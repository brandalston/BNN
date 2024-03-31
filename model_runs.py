from BNN import MIPBNN
import time, os, argparse, getopt, sys, math, csv, warnings
warnings.filterwarnings("ignore")
import UTILS, dataload
import numpy as np
from Benchmarks.Thorbjarnarson.mip.get_nn import get_nn
from Benchmarks.Thorbjarnarson.helper.misc import inference, infer_and_accuracy, clear_print, get_network_size,strip_network
from Benchmarks.Thorbjarnarson.helper.data import load_data, get_architecture
from Benchmarks.Thorbjarnarson.helper.fairness import equalized_odds, demographic_parity

example_skip_dict = {'s_138': 4, 's_15': 3, 's_89': 2, 's_42': 1, 's_0': 0}


def mip_run(argv):
    # print(argv)
    obj_func = None
    n_hidden_layers = 0
    examples_per_class = 1
    seed = 's_0'
    time_limit = 60
    model_eps = 0.001
    consol_log = 0
    file_out = None

    try:
        opts, args = getopt.getopt(argv, "o:h:c:t:s:e:z:f:",
                                   ["obj_func=", "n_hidden_layers=", "examples_per_class=", "time_limit=",
                                    "seed=", "model_eps=", "consol_log=", "results_file="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-o", "--obj_func"):
            obj_func = arg
        elif opt in ("-h", "--n_hidden_layers"):
            n_hidden_layers = arg
        elif opt in ("-c", "--examples_per_class"):
            examples_per_class = arg
        elif opt in ("-t", "--time_limit"):
            time_limit = int(arg)
        elif opt in ("-s", "--seed"):
            seed = arg
        elif opt in ("-e", "--model_eps"):
            model_eps = arg
        elif opt in ("-z", "--consol_log"):
            consol_log = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Train_Size','Rand_State','In-Acc','Out-Acc','Run_Time',
                       'Num_HL','Obj_Func','TL','Learning_Rate','TF_Seed']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = 'Seed:' + str(seed) + '_HL:' + str(n_hidden_layers) + '_Obj:' + str(obj_func) +\
                      'EC:'+str(examples_per_class) + '_TL:' + str(time_limit) + '.csv'
    else:
        output_name = file_out
    out_file = output_path + output_name
    if file_out is None:
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()

    """
    Run an experiment using a MIP-based approach.
    @params
        - obj_func: this indicates which MIP objective to use
            "weight": mip model using a min-weight
            "margin": mip model using a max-margin
        - n_hidden_layers: # of hidden layers in BNN
            - Fixed to 16
        - examples_per_class: # min number of examples per class during training
        - time_limit:
            - Time limit in minutes
        - examples_skip:
            - Used to construct different instance problems of the same size (without overlap)
            - We skip examples_skip*examples_per_class examples when creating the training set
            - Fixed to zero
        - model_eps:
            - Epsilon value used for max margin model
    """
    print("====================================")
    print(f"Obj: {obj_func}. Hidden Layers: {n_hidden_layers}. N: {10*examples_per_class}. Seed:", seed.replace("s_",""),".")
    print("====================================")
    # Configuration
    n_hidden_neurons = 16  # number of neurons per hidden layer
    n_threads = 1  # number of threads

    # Net architecture
    net = [28 * 28] + [n_hidden_neurons for _ in range(n_hidden_layers)] + [10]
    n_train = 10 * examples_per_class

    # """
    # loading the training set
    images, labels = dataload.mnist_train_per_class(examples_per_class, example_skip_dict[seed])
    labels = 2.0 * UTILS.get_one_hot_encoding(labels) - 1.0  # mapping labels to -1/1 vectors

    # Training the network
    BNN_model = MIPBNN(net, images, obj_func=obj_func)
    for i in range(n_train):
        BNN_model.add_example(images[i], labels[i], model_eps, show=False)
    print('Run Start: ' + str(time.strftime("%I:%M:%S %p", time.localtime())))
    BNN_model.optimize(time_limit, n_threads, consol_log=consol_log)
    print("-----------------------------")
    results = BNN_model.model_assign()

    # Testing the solution
    if results['found_sol']:
        weights, biases = BNN_model.get_weights()
        train_performance, test_performance = UTILS.model_acc(net, weights, biases, images, labels)
        results["rand_state"] = seed.replace("s_","")
        results["train_size"] = 10 * examples_per_class
        results["train_acc"] = train_performance
        results["test_acc"] = test_performance
        results["n_hidden_layers"] = n_hidden_layers
        results["obj_func"] = obj_func
        results['TL'] = time_limit
        results["weights"] = [w.tolist() for w in weights]
        results["biases"] = [b.tolist() for b in biases]

        print("Obj value: ", BNN_model.m.ObjVal)
        print("Obj bound: ", BNN_model.m.ObjBound)
        print("Gap:", BNN_model.m.MIPGap)
        print("Run time = %0.2f" % BNN_model.m.RunTime)
        print("Test performance = %0.3f" % test_performance)
        print("Train performance = %0.3f" % train_performance)
    else:
        print("TIMEOUT or INFEASIBLE!")

    # Saving the results
    UTILS.model_summary(results, out_file)
    print("-----------------------------\n")


def gd_run(argv):
    # print(argv)
    lr = 1e-3
    tf_seed = 0
    n_hidden_layers = 0
    examples_per_class = 1
    seed = 0
    time_limit = 60
    file_out = None
    model = None
    
    try:
        opts, args = getopt.getopt(argv, "m:r:h:t:i:c:s:f:",
                                   ["model=", "learning_rate=", "n_hidden_layers=", "time_limit=", 
                                    "tf_seed=", "examples_per_class=", "seed=", "results_file="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-m", "--model"):
            model = arg
        elif opt in ("-r", "--learning_rate"):
            lr = arg
        elif opt in ("-h", "--n_hidden_layers"):
            n_hidden_layers = arg
        elif opt in ("-t", "--time_limit"):
            time_limit = int(arg)
        elif opt in ("-i", "--tf_seed"):
            tf_seed = arg
        elif opt in ("-c", "--examples_per_class"):
            examples_per_class = arg
        elif opt in ("-s", "--seed"):
            seed = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Train_Size', 'Rand_State', 'In-Acc', 'Out-Acc', 'Run_Time',
                       'Num_HL', 'Obj_Func', 'TL', 'Learning_Rate', 'TF_Seed']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = 'Seed:' + str(seed) + '_HL:' + str(n_hidden_layers) + 'GD_LR:' + str(lr) + \
                      '_TFSeed:'+str(tf_seed) + '_EC:' + str(examples_per_class) + '_TL:' + str(time_limit) + '.csv'
    else:
        output_name = file_out
    out_file = output_path + output_name
    if file_out is None:
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()
    """
    Run an experiment using a gradient descent approach using ternary weights (-1,0,1).
    @params
        - lr: Learning rate
        - tf_seed: Random seed used to initialize the weights of the network
        - n_hidden_layers: # of hidden layers
            - Fixed to 16
        - examples_per_class: Min # number of examples per class during training
        - time_limit:
            - Time limit in minutes
        - examples_skip:
            - Used to construct different instance problems of the same size (without overlap)
            - We skip examples_skip*examples_per_class examples when creating the training set
            - Fixed to zero
    """

    print("====================================")
    print(f"Obj: GD-{model}. Hidden Layers: {n_hidden_layers}. N: {10 * examples_per_class}. Seed:",
          seed.replace("s_", ""), f". TF_Seed: {tf_seed}. LR: {lr}")
    print("====================================")

    # Setting the network's architecture
    n_input_neurons = 28 * 28
    n_hidden_neurons = 16
    n_output_neurons = 10
    net = [28 * 28] + [n_hidden_neurons for _ in range(n_hidden_layers)] + [10]

    # loading the training set
    train_data, train_labels = dataload.mnist_train_per_class(examples_per_class, example_skip_dict[seed])
    train_labels = UTILS.get_one_hot_encoding(train_labels)

    # Training and testing the net
    from Benchmarks.ICARTE import StandardNeuralNet
    if 'binary' == model:
        nn = StandardNeuralNet(n_input_neurons, n_hidden_neurons, n_hidden_layers, n_output_neurons, lr, tf_seed, False)
    else:
        nn = StandardNeuralNet(n_input_neurons, n_hidden_neurons, n_hidden_layers, n_output_neurons, lr, tf_seed, True)
    print('Run Start: ' + str(time.strftime("%I:%M:%S %p", time.localtime())))
    start = time.perf_counter()
    is_sat = nn.train(train_data, train_labels, train_data, train_labels, time_limit)
    run_time = (time.perf_counter() - start)

    # Testing the solution (weights of dead neurons are set to zero here)
    weights, biases = nn.get_weights()
    dead_inputs = np.all(train_data == train_data[0, :], axis=0)
    for neuron_in in range(n_input_neurons):
        if dead_inputs[neuron_in]: weights[0][neuron_in, :] = np.zeros(net[1])
    train_performance, test_performance = UTILS.model_acc(net, weights, biases, train_data, 2 * train_labels - 1)
    print("-----------------------------")
    print("Run time = %0.3f" % run_time)
    print("Train acc = %0.3f" % train_performance)
    print("Test acc = %0.3f" % test_performance)
    print("-----------------------------\n")

    # saving results
    results = {}
    results['rand_state'] = seed.replace("s_","")
    results['train_size'] = 10 * examples_per_class
    results["train_acc"] = train_performance
    results["test_acc"] = test_performance
    results["run_time"] = run_time
    results['n_hidden_layers'] = n_hidden_layers
    results["obj_func"] = "GD-"+model
    results["learning_rate"] = lr
    results["tf_seed"] = tf_seed
    results['TL'] = time_limit
    results["is_sat"] = (train_performance == 1.0)
    results["weights"] = [w.tolist() for w in weights]
    results["biases"] = [b.tolist() for b in biases]
    UTILS.model_summary(results, out_file)

    # close the network session
    nn.close()


def thor_run(argv):
    # print(argv)
    obj_func = None
    n_hidden_layers = 0
    examples_per_class = 1
    seed = 0
    time_limit = 60
    consol_log = 0
    bound = 1
    file_out = None

    try:
        opts, args = getopt.getopt(argv, "o:h:c:t:s:b:z:f:",
                                   ["obj_func=", "n_hidden_layers=", "examples_per_class=", "time_limit=",
                                    "seed=", "bound=", "consol_log=", "results_file="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-o", "--obj_func"):
            obj_func = arg
        elif opt in ("-h", "--n_hidden_layers"):
            n_hidden_layers = arg
        elif opt in ("-c", "--examples_per_class"):
            examples_per_class = arg
        elif opt in ("-t", "--time_limit"):
            time_limit = int(arg)
        elif opt in ("-s", "--seed"):
            seed = arg
        elif opt in ("-b", "--bound"):
            bound = arg
        elif opt in ("-z", "--consol_log"):
            consol_log = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg
    ''' Columns of the results file generated '''
    summary_columns = ['Train_Size', 'Rand_State', 'In-Acc', 'Out-Acc', 'Run_Time',
                       'Num_HL', 'Obj_Func', 'TL', 'Learning_Rate', 'TF_Seed']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = 'Seed:' + str(seed) + '_HL:' + str(n_hidden_layers) + '_Obj:' + str(obj_func) + \
                      'EC:' + str(examples_per_class) + '_TL:' + str(time_limit) + '.csv'
    else:
        output_name = file_out
    out_file = output_path + output_name
    if file_out is None:
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()

    N = examples_per_class
    focus = 2
    train_time = time_limit
    loss = obj_func
    reg = False
    fair = False

    data = load_data("mnist", N, seed)

    hls = [16]*n_hidden_layers if n_hidden_layers > 0 else []
    architecture = get_architecture(data, hls)
    # print("architecture", architecture)
    print("====================================")
    print(f"Obj: {loss}. Hidden Layers: {n_hidden_layers}. N: {10*examples_per_class}. Bound: {bound}. Seed: {seed}.")
    print("====================================")
    nn = get_nn(loss, data, architecture, bound, reg, fair)
    print('Run Start: ' + str(time.strftime("%I:%M:%S %p", time.localtime())))
    nn.train(train_time * 60, focus, consol_log=consol_log)
    print("-----------------------------")

    print("Obj value: ",  nn.get_objective())
    print("Obj bound: ", nn.get_bound())
    print("Gap:", nn.get_gap())

    varMatrices = nn.extract_values()

    train_acc = infer_and_accuracy(nn.data['train_x'], nn.data["train_y"], varMatrices, nn.architecture)
    # val_acc = infer_and_accuracy(nn.data['val_x'], nn.data["val_y"], varMatrices, nn.architecture)
    test_acc = infer_and_accuracy(nn.data['test_x'], nn.data["test_y"], varMatrices, nn.architecture)

    print("Run time = %0.2f" % nn.m.RunTime)
    print("Test acc = %0.3f" % test_acc)
    print("Train acc = %0.3f" % train_acc)

    results = {}
    results['rand_state'] = seed
    results['train_size'] = 10 * examples_per_class
    results["train_acc"] = train_acc
    results["test_acc"] = test_acc
    results["run_time"] = nn.m.RunTime
    results['n_hidden_layers'] = n_hidden_layers
    results["obj_func"] = "THOR-"+obj_func
    results['MIPGap'] = nn.get_gap()
    results['ObjBound'] = nn.get_bound()
    results['ObjVal'] = nn.get_objective()
    results["learning_rate"] = 'N/A'
    results["tf_seed"] = 'N/A'
    results['TL'] = time_limit
    results["is_sat"] = (train_acc == 1.0)
    # results["weights"] = [w.tolist() for w in weights]
    # results["biases"] = [b.tolist() for b in biases]
    UTILS.model_summary(results, out_file)
    print("-----------------------------\n")
