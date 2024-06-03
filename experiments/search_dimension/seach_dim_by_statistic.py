# Description: This script is used to search the dimensions of hidden states that are more likely to be positional hidden states.

import numpy as np
import scipy
from tqdm import tqdm


def fit_and_monotic(data, degree=3, skip=50):
    data = data[skip:]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # fit
    x = np.arange(len(data), dtype=np.float32)
    z1 = np.polyfit(x, data.astype(np.float32), degree)
    p1 = np.poly1d(z1)
    y_fit = p1(x)
    # determine the monotonicity, i.e., whether the difference is of the same sign
    y_fit_diff = np.diff(y_fit)

    if np.all(y_fit_diff >= 0):
        return 1
    elif np.all(y_fit_diff <= 0):
        return 2
    else:
        return 0


# for each dimension, calculate the number of increasing or decreasing layers
def get_mono_layers(hidden_states_dim: np.ndarray, degree=3, skip=100) -> list[int]:
    inc_layers = []
    dec_layers = []
    for layer in range(2, hidden_states_dim.shape[0] - 1):
        hidden_states_dim_layer = hidden_states_dim[layer]
        m = fit_and_monotic(hidden_states_dim_layer, degree=degree, skip=skip)
        if m == 1:
            inc_layers.append(layer)
        elif m == 2:
            dec_layers.append(layer)

    # determine whether there are more increasing layers or decreasing layers
    inc_layer_num = len(inc_layers)
    dec_layer_num = len(dec_layers)
    if inc_layer_num > dec_layer_num:
        mono_layers = inc_layers
    else:
        mono_layers = dec_layers

    return mono_layers


# the smoothness of the curve is defined as the integral of the square of the second derivative
def get_smoothness(x):
    # normalize to 0-1
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    # smooth with a sliding window
    x = scipy.signal.savgol_filter(x, window_length=10, polyorder=3, axis=0)
    # calculate the first derivative, i.e., the first-order difference
    dx = np.diff(x, axis=0)
    # calculate the second derivative, i.e., the second-order difference
    ddx = np.diff(dx, axis=0)
    # calculate the square
    ddx2 = np.square(ddx)
    # calculate the integral
    smoothness = np.sum(ddx2, axis=0)
    return smoothness


# find the smallest smoothness in the monotonous layers
def get_min_smoothness_across_layers(hidden_states_all_mean, layers, skip=300):
    smoothness_layers = [get_smoothness(hidden_states_all_mean[i][skip:]) for i in layers]
    return np.min(smoothness_layers)


def search_by_statistic(hidden_states_path, topk) -> list[int]:
    if isinstance(hidden_states_path, np.ndarray):
        hidden_states = hidden_states_path
    else:
        hidden_states = np.load(hidden_states_path).astype(np.float32)  # shape=(num_layers,num_samples,hidden_size)

    if len(hidden_states.shape) == 4:
        # if shape is (num_layers,num_heads,seq_len,head_dim)，reshape to (num_layers,seq_len,hidden_size)
        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(hidden_states.shape[0],
                                                                    hidden_states.shape[2],
                                                                    -1)

    num_layers = hidden_states.shape[0]
    hidden_size = hidden_states.shape[2]
    seq_len = hidden_states.shape[1]

    # discard the embedding layer
    if num_layers % 2 == 1:
        hidden_states = hidden_states[1:]
        num_layers = num_layers - 1

    # traverse each dimension, calculate the number of increasing or decreasing layers (take the maximum value)
    each_dim_mono_layers = {dim: get_mono_layers(hidden_states[:, :, dim], skip=300) for dim in
                            tqdm(range(hidden_size), desc="eval each dimension's monotonicity")}

    # keep the dimensions with monotonic layers greater than num_layers // 4
    dim_cand1 = [dim for dim in range(hidden_size) if len(each_dim_mono_layers[dim]) > num_layers // 4]
    dim_cand1 = np.array(dim_cand1).astype(np.int32)

    # calculate the smoothness of each dimension, take the top-k layers with the smallest smoothness
    dim_cand_smoothness = [
        get_min_smoothness_across_layers(hidden_states[:, :, dim], each_dim_mono_layers[dim], skip=200) for dim in
        dim_cand1]
    dim_cand = dim_cand1[np.argsort(dim_cand_smoothness)[:int(topk)]]
    dim_cand = dim_cand.tolist()
    dim_cand = [int(i) for i in dim_cand]

    print(
        "positional dimensions are {}".format(str(dim_cand)))

    return dim_cand


if __name__ == '__main__':
    save_path_hidden = "./hidden_states/hidden_states_of_llama2_7b.npy"
    result = search_by_statistic(save_path_hidden, topk=10)
