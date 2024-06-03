from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pathlib

def draw_plot(state, dim, save_dir, skip_first=30,num_cols=6):
    num_layers = state.shape[0]
    import math
    num_rows = math.ceil(num_layers/num_cols)
    plt.figure(figsize=(6*num_cols, 6*num_rows))

    for layer_index in tqdm(range(state.shape[0]), desc="visualize" + " dim=" + str(dim)):
        plt.subplot(num_rows, num_cols, layer_index + 1)
        states_layer = state[layer_index]
        # transpose to (hidden_size,seq_len)
        states_layer = states_layer.T
        # select dimension
        states_layer = states_layer[dim]
        # draw
        plt.plot(states_layer[skip_first:])

        if num_layers % 2 == 0:
            plt.title("layer " + (str(layer_index)), fontsize=30)
        else:
            plt.title("layer " + (str(layer_index - 1) if layer_index > 0 else "embedding"), fontsize=30)

        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)

    plt.tight_layout()
    plt.savefig(save_dir + "/hidden_dim=" + str(dim)+ ".png")
    print("save to", save_dir + "/hidden_dim=" + str(dim)+ ".png")
    plt.show()
    plt.close()

def visualize_hidden(hidden_save_path, dim, output_dir, skip_first, discard_embed=True):
    hidden_states=np.load(hidden_save_path)
    if discard_embed:
        #discard embedding layer
        hidden_states=hidden_states[1:]

    print("num layers",hidden_states.shape[0])
    print("seq len",hidden_states.shape[1])
    print("hidden size",hidden_states.shape[2])

    #visualize
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    draw_plot(hidden_states, dim, output_dir,skip_first)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_states_path", type=str, required=True, help="hidden states save path")
    parser.add_argument("--dim", type=int, required=True, help="dimension to visualize")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir")
    parser.add_argument("--skip_first", type=int, default=30, help="skip the first n tokens")
    args = parser.parse_args()

    visualize_hidden(args.hidden_states_path,args.dim,args.output_dir,args.skip_first)


