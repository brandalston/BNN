"""
This is a MIP model that aims to find the BNNs with fewer non-zero weights that fits the training set.
"""
import random, time, math, warnings
warnings.filterwarnings("ignore")
from gurobipy import *
import numpy as np


class MIPBNN:

    def __init__(self, architecture, data, obj_func='bias', model_eps=0.001):
        """
        "architecture" is a list of numbers indicating how many neurons each layer has
          e.g. [2,2,1] -> 2 input neurons, then 2 neurons on a hidden layer, and one output neuron
        """
        self.layers = architecture
        self.obj = obj_func
        self.data = data
        self.model_eps = model_eps
        self.B = None
        self.W = None
        self.C = None
        self.M = None
        self.indicator = None

        self.m = Model("MIP_BNN")

        # Removing dead inputs
        dead_inputs = np.all(data == data[0, :], axis=0)

        # Weights and biases
        self.weights = {}
        self.biases = {}
        self.margins = {}
        for layer_id in range(1, len(self.layers)):
            for neuron_out in range(self.layers[layer_id]):
                # Add weights
                for neuron_in in range(self.layers[layer_id - 1]):
                    # layer_id: layer of the output neuron
                    w_id = (neuron_in, layer_id, neuron_out)
                    if layer_id == 1 and dead_inputs[neuron_in]:
                        w = 0
                    else:
                        w = self.m.addVar(vtype=GRB.INTEGER, name="W%d_%d-%d" % w_id, lb=-1, ub=1)
                    self.weights[w_id] = w
                # Add bias of output neuron
                b_id = (layer_id, neuron_out)
                self.B = self.m.addVar(vtype=GRB.INTEGER, name="b_%d-%d" % b_id, lb=-1, ub=1)
                self.biases[b_id] = self.B
                # Add margins per neuron if maximizing margins
                if 'margin' in self.obj:
                    n_id = (layer_id, neuron_out)
                    self.margins[n_id] = self.m.addVar(vtype=GRB.CONTINUOUS, name="m%d-%d" % n_id, lb=0)
        # Max margin loss
        if 'margin' in self.obj:
            self.loss = sum(list(self.margins.values()))
            self.eg_id = 0
            self.activations = {}

        # Min weights loss
        if 'bias' in self.obj:
            w_abs = []
            for w_id in self.weights:
                if type(self.weights[w_id]) is int:
                    continue
                self.W = self.m.addVar(vtype=GRB.BINARY, name="aw%d_%d-%d" % w_id)
                self.m.addConstr(self.weights[w_id] <= self.W)
                self.m.addConstr(self.weights[w_id] >= -self.W)
                w_abs.append(self.W)
            for b_id in self.biases:
                self.B = self.m.addVar(vtype=GRB.BINARY, name="ab_%d-%d" % b_id)
                self.m.addConstr(self.biases[b_id] <= self.B)
                self.m.addConstr(self.biases[b_id] >= -self.B)
                w_abs.append(self.W)
            self.loss = sum(w_abs)

            # No loss function
            self.eg_id = 0
            self.activations = {}

        # pass to model for callback purposes
        self.m._data = self.data
        self.m._biases = self.biases
        self.m._weights = self.weights
        self.m._margins = self.margins
        self.m._cbnum, self.m._cbcuts = 0, 0

    def _add_neuron_weight_binding(self, u_indicator, w, n_in, layer_id, n_out):
        C = self.m.addVar(vtype=GRB.CONTINUOUS, name="I_%d-%d-%d_%d" % (n_in, layer_id, n_out, self.eg_id), lb=-1, ub=1)
        self.m.addConstr(C - w + 2 * u_indicator <= 2)
        self.m.addConstr(C + w - 2 * u_indicator <= 0)
        self.m.addConstr(C - w - 2 * u_indicator >= -2)
        self.m.addConstr(C + w + 2 * u_indicator >= 0)
        return C

    def add_example(self, data, label, eps, show=False):
        """
        NOTE:
            - the neurons are binary variables (0,1)
            - however, the '0' value has to be mapped to '-1' when adding the constraints (i.e. replace 'n' by '2*n-1')
        """
        # Adding the layers
        neurons = {}
        for layer_id in range(1, len(self.layers)):
            for n_out in range(self.layers[layer_id]):
                # FIRST MUST COMPUTE EACH NEURON'S PREACTIVATION VALUE FROM DV'S
                if layer_id == 1:
                    # First hidden layer neuron
                    pre_activation = sum([data[i] * self.weights[(i, 1, n_out)] for i in range(len(data))])
                else:
                    inputs = []
                    for n_in in range(self.layers[layer_id - 1]):
                        indicator = neurons[(layer_id - 1, n_in)]
                        w = self.weights[(n_in, layer_id, n_out)]
                        # Add preactivation constrains between layers
                        C = self._add_neuron_weight_binding(indicator, w, n_in, layer_id, n_out)
                        inputs.append(C)
                    pre_activation = sum(inputs)
                # Update W/ bias
                pre_activation += self.biases[(layer_id, n_out)]
                # or W/ neuron's margin depending on obj. func.
                if 'margin' in self.obj:
                    margin = self.margins[(layer_id, n_out)]
                # Neuron activations
                if layer_id == len(self.layers) - 1:
                    # At an output neuron
                    if label[n_out] > 0:
                        if 'bias' in self.obj: self.m.addConstr(pre_activation >= 0)
                        if 'margin' in self.obj: self.m.addConstr(pre_activation >= margin)
                    else:
                        if 'bias' in self.obj: self.m.addConstr(pre_activation <= -1)
                        if 'margin' in self.obj: self.m.addConstr(pre_activation <= -margin - eps)
                else:
                    # At Hidden neuron
                    self.indicator = self.m.addVar(vtype=GRB.BINARY, name="n%d-%d_%d" % (layer_id, n_out, self.eg_id))
                    # Indicator constraint version instead of callback cuts
                    if 'bias-indicator' == self.obj:
                        self.m.addConstr((self.indicator == 1) >> (pre_activation >= 0))
                        self.m.addConstr((self.indicator == 0) >> (pre_activation <= -1))
                    if 'margin-indicator' == self.obj:
                        self.m.addConstr((self.indicator == 1) >> (pre_activation >= margin))
                        self.m.addConstr((self.indicator == 0) >> (pre_activation <= -margin - self.model_eps))

                    # neuron id of neuron 'n': (layer_id th layer, n_out th neuron in layer)
                    neurons[(layer_id, n_out)] = self.indicator
                    self.activations[(layer_id, n_out, self.eg_id)] = self.indicator

        # Track training point ID for auxiliary variables
        self.eg_id += 1
        # pass back to model for callback purposes
        self.m._indicator = self.indicator

    def optimize(self, time_limit, n_threads=1, consol_log=0):
        """
        Returns True if no feasible solution exists
        """

        # Params
        # self.m.Params.OutputFlag = 0
        self.m.Params.LogToConsole = consol_log
        self.m.Params.TimeLimit = time_limit * 60
        self.m.Params.Threads = n_threads

        # Optimize
        if 'bias' in self.obj: self.m.setObjective(self.loss, GRB.MINIMIZE)
        if 'margin' in self.obj: self.m.setObjective(self.loss, GRB.MAXIMIZE)
        self.m.update()
        print('Optimizing model')
        self.m.optimize()
        # self.m.optimize(MIPBNN.callback)

        # Is feasible?
        return self.m.SolCount > 0

    @staticmethod
    def callback(model, where):
        # Model Termination
        if where == GRB.Callback.MIP:
            if abs(model.cbGet(GRB.Callback.MIP_OBJBST) -
                   model.cbGet(GRB.Callback.MIP_OBJBND)) < model.Params.FeasibilityTol:
                model.terminate()

        # Verify training point has reached neuron through valid previous activations
        if where == GRB.Callback.MIPSOL:
            model._cbnum += 1
            start = time.perf_counter()
            for indicator in model._data:
                pass
        '''sum over W instead of u_indicator'''

    # Warm start model
    def warm_start(self, weights, biases, activations):
        # weights and biases
        for layer_id in range(1, len(self.layers)):
            for neuron_out in range(self.layers[layer_id]):
                # Adding weights
                for neuron_in in range(self.layers[layer_id - 1]):
                    w = self.weights[(neuron_in, layer_id, neuron_out)]
                    if not (type(w) is int):
                        w.start = weights[layer_id - 1][neuron_in, neuron_out]
                # Adding biases
                b = self.biases[(layer_id, neuron_out)]
                b.start = biases[layer_id - 1][neuron_out]
        # activations
        for layer_id in range(1, len(self.layers) - 1):
            for eg_id in range(self.eg_id):
                for n_out in range(self.layers[layer_id]):
                    n = self.activations[(layer_id, n_out, eg_id)]
                    n.start = activations[layer_id][eg_id, n_out]

    def model_assign(self):
        model_results = {}
        model_results["found_sol"] = self.m.SolCount > 0
        model_results["ObjVal"] = self.m.ObjVal
        model_results["ObjBound"] = self.m.ObjBound
        model_results["MIPGap"] = self.m.MIPGap
        model_results["is_optimal"] = (self.m.status == GRB.OPTIMAL)
        model_results["bb_nodes"] = self.m.NodeCount
        model_results["num_vars"] = self.m.NumIntVars + self.m.NumBinVars
        model_results["cb_count"] = self.m._cbnum
        model_results["cb_cuts"] = self.m._cbcuts
        model_results['run_time'] = self.m.RunTime
        model_results['learning_rate'] = 'N/A'
        model_results['tf_seed'] = 'N/A'

        """
        if self.m.SolCount > 0:
            print("objective: %0.2f" % model_results["objective"])
            print("bound: %0.2f" % model_results["bound"])
            print("gap: %0.2f" % model_results["gap"])
        """

        return model_results

    def get_weights(self):
        # Returns the best weights found so far
        w_ret, b_ret = [], []
        for layer_id in range(1, len(self.layers)):
            n_in = self.layers[layer_id - 1]
            n_out = self.layers[layer_id]
            weights = np.zeros((n_in, n_out))
            biases = np.zeros((n_out,))
            for j in range(n_out):
                for i in range(n_in):
                    w_id = (i, layer_id, j)
                    w = self.weights[w_id]
                    if type(w) is int:
                        weights[i, j] = 0
                    else:
                        weights[i, j] = w.X
                biases[j] = self.biases[(layer_id, j)].X
            w_ret.append(weights)
            b_ret.append(biases)

        return w_ret, b_ret
