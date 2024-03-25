import model_runs, dataload, UTILS

n_hidden_layers = [1, 2]
examples_per_class = [1,5,10]
rand_states = [0, 15, 42, 89, 138]
time_limit = 30
learning_rate = [1e-3,1e-4,1e-5,1e-6]
tf_seed = [66,56,16]

# GD
for ec in examples_per_class:
    for lr in learning_rate:
        for tfs in tf_seed:
            model_runs.gd_run(["-h", 0, "-s", f's_{15}', "-r", lr, "-c", ec, "-t", time_limit, "-i", tfs, "-f", 'bnn_runs.csv'])
