import numpy as np
from gurobipy import Model, GRB, quicksum
import copy


def bound_prop(weight, bias, cnt_pre, operator_type, act_func):
    if operator_type == 'Dense':
        n_neu = weight.shape[-1]
        num_cols = 2 if act_func == 'relu' else 1
        cnt = np.zeros([n_neu, num_cols])
        cnt[:, 0] = np.sum(weight * np.expand_dims(cnt_pre[:, -1], axis=1), axis=0) + bias

        oas = get_status(cnt, operator_type, act_func) if act_func == 'relu' else []
        if act_func == 'relu':
            cnt[:, 1] = np.maximum(0, cnt[:, 0])
    return cnt, oas


def get_status(cnt, layer_type, layer_activation):
    if layer_activation == 'relu':
        if layer_type == 'Dense':
            oas = np.asarray(np.sign(cnt[:, 0]))
            oas[np.where(cnt[:, 0] == 0)] = -1
    return oas


def net_propagate(k_start, W, layer_type, layer_activation, center, oas):
    n_layers = len(layer_type)
    for i in range(k_start, n_layers + 1):
        if layer_type[i] == 'Dense' and i != n_layers + 1:
            cnt, oas[i] = bound_prop(W[i][0], W[i][1], center[i - 1], layer_type[i], layer_activation[i])
        center[i] = np.asarray(cnt)
    if k_start == 1:
        gb_inds = {
            i + 1: np.arange(np.size(W[i + 1][0])).reshape(np.shape(W[i + 1][0]), order='F') if layer_type[i + 1] == 'Dense' else {}
            for i in range(n_layers)
        }
        return center, oas, gb_inds
    return center, oas


def model_generator(ii, W, center, layer_type, gb_inds, gb_model, epsilon):
    keysList = list(center.keys())
    key_max = np.max(np.asarray(keysList))
    key_min = np.min(np.asarray(keysList))
    if layer_type == 'Dense':
        n_out = np.shape(center[key_max])[0]
        n_in = np.shape(center[key_min])[0]
        n_weights = (n_in + 1) * n_out
    if ii == 0:
        model = Model()
        variables = model.addVars(int(n_weights), lb=-1 * float('inf'), name="variables")
        model.Params.LogToConsole = 1
        model.Params.OutputFlag = 1
    else:
        model = gb_model.copy()
        variables = model.getVars()
    if layer_type == 'Dense':
        for m in range(n_out):
            ind_list = np.squeeze(gb_inds[:, m].reshape((-1, 1), order='F'))
            if center[key_max][m, 0] >= 0:
                model.addConstr(quicksum(
                    variables[ind_list[z]] * center[key_min][z, -1] for z in range(n_in)) + variables[
                                    np.prod(W[0].shape) + m] >= 0)
                model.addConstr(quicksum(
                    variables[ind_list[z]] * center[key_min][z, -1] for z in range(n_in)) + variables[
                                    np.prod(W[0].shape) + m] >= (1 - epsilon) * center[key_max][m, 0])
                model.addConstr(quicksum(
                    variables[ind_list[z]] * center[key_min][z, -1] for z in range(n_in)) + variables[
                                    np.prod(W[0].shape) + m] <= (1 + epsilon) * center[key_max][m, 0])
            else:
                model.addConstr(quicksum(
                    variables[ind_list[z]] * center[key_min][z, -1] for z in range(n_in)) + variables[
                                    np.prod(W[0].shape) + m] <= 0)
                model.addConstr(quicksum(
                    variables[ind_list[z]] * center[key_min][z, -1] for z in range(n_in)) + variables[
                                    np.prod(W[0].shape) + m] >= (1 + epsilon) * center[key_max][m, 0])
                model.addConstr(quicksum(
                    variables[ind_list[z]] * center[key_min][z, -1] for z in range(n_in)) + variables[
                                    np.prod(W[0].shape) + m] <= (1 - epsilon) * center[key_max][m, 0])
        model.update()
    return model


def weight_opt(gb_model):
    model = gb_model.copy()
    variables = model.getVars()
    tmp = model.addVars(gb_model.NumVars, lb=-1 * float('inf'), name="tmp")
    obj = (quicksum(tmp[z] for z in range(gb_model.NumVars)))
    for i in range(gb_model.NumVars):
        model.addConstr(variables[i] <= tmp[i])
        model.addConstr(-variables[i] <= tmp[i])
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    if model.status != 2:
        model.Params.DualReductions = 0
        model.reset()
        model.optimize()
        if model.status != 2:
            model.feasRelaxS(1, False, False, True)
            model.reset()
            model.optimize()
    all_vars = model.getVars()
    values = model.getAttr("X", all_vars)
    return values


def save_inds(layer_activation):
    act_inds = [k for k, v in layer_activation.items() if v == 'relu']
    k_save = act_inds[1:] + [len(layer_activation)]
    return act_inds, k_save


def update_model_weights(layer_activation, layer_type, gb_model, weight_opt, W, model):
    weights = dict()
    count_relu = 0
    count_dense_relu = 0
    dense_relu_ind = []
    relu_ind = []
    for i in layer_activation:
        if layer_activation[i] == 'relu':
            count_relu += 1
            relu_ind.append(i)
        if layer_type[i] == 'Dense' and layer_activation[i] == 'relu':
            count_dense_relu += 1
            dense_relu_ind.append(i)
    count_range = list(range(1, count_relu + 1))
    for j, i in enumerate(gb_model):
        if gb_model[i] is not None:
            weights[dense_relu_ind[j]] = weight_opt(gb_model[count_range[j]])
    keysList = list(weights.keys())
    for j, i in enumerate(keysList):
        ll = len(weights[i]) // 2
        tmp = weights[i][:ll]
        ind = dense_relu_ind[j]
        W_new = copy.deepcopy(W)
        W_new[ind][0] = np.reshape(tmp[:np.prod(W[ind][0].shape)], np.shape(W[ind][0]), order='F')
        W_new[ind][1] = tmp[-np.prod(W[ind][1].shape):]
        k = count_range[j]
        model.weights[2 * (k - 1)].assign(W_new[ind][0])
        model.weights[2 * (k - 1) + 1].assign(W_new[ind][1])