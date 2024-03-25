import model_runs, dataload, UTILS, warnings
warnings.filterwarnings("ignore")

"""
images, labels = dataload.mnist_train_per_class(10, 0)
labels = 2.0 * UTILS.get_one_hot_encoding(labels) - 1.0  # mapping labels to -1/1 vectors

print(len(images), images.shape)
for i in range(5):
    print(images[i],'\n', type(images[i]))
"""
n_hidden_layers = [0, 1, 2]
examples_per_class = [1,5,10]
rand_states = [0, 15, 42, 89, 138]
time_limit = 60
learning_rate = [1e-3,1e-4,1e-5,1e-6]
tf_seed = [66,56,16]

# model_runs.gd_run(["-h", 0, "-s", f's_{0}', "-r", 1e-3, "-c", 2, "-t", 2, "-i", 42, "-f", 'test_dump.csv'])
model_runs.mip_run(["-o", 'margin-indicator', "-h", 0, "-c", 5, "-t", 10, "-s", f's_{138}', "-e", 0.0001, "-z", 1, "-f", 'test_dump.csv'])
