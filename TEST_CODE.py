import model_runs, dataload, UTILS, warnings, csv
warnings.filterwarnings("ignore")

"""
images, labels = dataload.mnist_train_per_class(10, 0)
labels = 2.0 * UTILS.get_one_hot_encoding(labels) - 1.0  # mapping labels to -1/1 vectors

print(len(images), images.shape)
for i in range(5):
    print(images[i],'\n', type(images[i]))
"""
n_hidden_layers = [1, 2]
examples_per_class = [1,2,5,8,10]
rand_states = [0, 15, 42, 89, 138]
time_limit = 60
learning_rate = [1e-3,1e-4,1e-5,1e-6]
tf_seed = [66,56,16]

# model_runs.gd_run(["-h", 0, "-s", f's_{0}', "-r", 1e-3, "-c", 2, "-t", 2, "-i", 42, "-f", 'test_dump.csv'])
# model_runs.mip_run(["-o", 'margin-indicator', "-h", 0, "-c", 5, "-t", 10, "-s", f's_{138}', "-e", 0.0001, "-z", 1, "-f", 'test_dump.csv'])

# MIP
obj_func = ['bias-indicator', 'margin-indicator']#, 'bias', 'margin']
model_eps = 0.001
for obj in ['bias-indicator','margin-indicator']:
    for seed in rand_states:
        for nhl in n_hidden_layers:
            for ec in examples_per_class:
                try:
                    model_runs.mip_run(["-o", obj, "-h", nhl, "-c", ec, "-t", time_limit,
                                        "-s", f's_{seed}', "-e", 0.001, "-z", 0, "-f", 'test_dump.csv'])
                except:
                    print(f'Obj: {obj}, HL: {nhl}, EC: {ec} INFEASIBLE!!')
                    with open('results_files/test_dump.csv', mode='a') as results:
                        results_writer = csv.writer(results, delimiter=',', quotechar='"')
                        results_writer.writerow([
                            0, 10*ec, 'N/A', 'N/A', 'N/A', nhl, obj_func,
                            'N/A', 'N/A', time_limit, 'N/A', 'N/A', 'N/A'])
                        results.close()
