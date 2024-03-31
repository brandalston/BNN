import model_runs, dataload, UTILS

n_hidden_layers = [1, 2]
#examples_per_class = [1, 2, 5, 8, 10]
examples_per_class = [3,4,6,7,9]
rand_states = [0, 15, 42, 89, 138]
time_limit = 30
learning_rate = [1e-3, 1e-4, 1e-5, 1e-6]
tf_seed = [66, 56, 16]

# model_runs.mip_run(["-o", 'bias-indicator', "-h", 0, "-c", 2, "-t", time_limit,
#                    "-s", f's_{89}', "-e", 0.001, "-z", 0, "-f", 'bnn_runs.csv'])


# MIP
obj_func = ['bias-indicator', 'margin-indicator', 'bias', 'margin']
model_eps = 0.001
for ec in [1,2,3,4,5,6,7,8,9,10]:
    model_runs.mip_run(["-o", 'bias-indicator', "-h", 0, "-c", ec, "-t", time_limit,
                        "-s", f's_{42}', "-e", 0.001, "-z", 0, "-f", 'bnn_runs.csv'])
