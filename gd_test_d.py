import model_runs, dataload, UTILS

n_hidden_layers = [0, 1, 2]
examples_per_class = [3,4,6,7,9]
rand_states = [0, 15, 42, 89, 138]
time_limit = 30
learning_rate = [1e-3,1e-4,1e-5,1e-6]
tf_seed = [66,56,16]

# GD
for ec in examples_per_class:
    for lr in learning_rate:
        for tfs in tf_seed:
            model_runs.gd_run(["-m", 'GD-binary', "-h", 0, "-s", f's_{89}', "-r", lr, "-c", ec, "-t", time_limit, "-i", tfs, "-f", 'bnn_runs.csv'])


examples_per_class = [1,2,3,4,5,6,7,8,9,10]
# GD
for ec in examples_per_class:
    for lr in learning_rate:
        for tfs in tf_seed:
            model_runs.gd_run(["-m", 'GD-ternary', "-h", 0, "-s", f's_{89}', "-r", lr, "-c", ec, "-t", time_limit, "-i", tfs, "-f", 'bnn_runs.csv'])
