from pathlib import Path
from scipy.io import savemat, loadmat
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import os

project_path = Path(__file__).parent.parent
fig_path = project_path / "saved\\figs\\"


def plot_and_save_figs(bias, output, target, sums, partial_change=None, layer_change_lim=None):
    save_b = bias.detach().cpu().numpy()
    save_out = output.detach().cpu().numpy()
    save_target = target.detach().cpu().numpy()
    save_sum = sums.detach().cpu().numpy()
    savemat(fig_path / "LayerMasking" / f'results.mat',
            {'bias': save_b, 'output': save_out, 'target': save_target, 'sum': save_sum})  # Only EN

    fig, axs = plt.subplots(2, 3)
    # fig.suptitle(f'{MODEL_NAME}_{method}_{i}_{epochs} \n X std_mean {x} \n Y std_mean: {y}', fontsize=16)
    size = 7
    ax0 = axs[0][0]
    ax0.hist(save_b[:, 0], bins=len(save_b), color='b')
    ax0.axis('equal')
    ax0.set_title("Energy Bias hist", size=size)

    ax1 = axs[0][1]
    ax1.hist(save_target[:, 0], bins=len(save_target), color='b')
    ax1.axis('equal')
    ax1.set_title("True EN hist", size=size)

    ax2 = axs[0][2]
    ax2.hist(save_out[:, 0], bins=len(save_out), color='b')
    ax2.axis('equal')
    ax2.set_title("Pred EN hist", size=size)

    ax3 = axs[1][0]
    ax3.scatter(save_sum, save_target[:, 0])
    m, b = np.polyfit(save_sum, save_target[:, 0], 1)
    ax3.axis('equal', xmin=-0.5, xmax=1, ymin=min(save_target), ymax=max(save_target))
    ax3.set_title(f"sum over pred,\nfit: m: {m:2f}, b: {b:2f}", size=size)

    ax4 = axs[1][1]
    m, b = np.polyfit(save_sum, save_out[:, 0], 1)
    ax4.scatter(save_sum, save_out[:, 0])
    ax4.axis('equal', xmin=min(save_sum), xmax=max(save_sum), ymin=min(save_out), ymax=max(save_out))
    ax4.set_title(f"sum over true,\nfit: m: {m:2f}, b: {b:2f}", size=size)

    plt.tight_layout()
    plt.savefig(fig_path / f'results.png')
    plt.close(fig)
    return


def load_predict_energies(path, percentage, layer_masking_lim):
    #   Loads results, splits by energy ranges, calculates std and plots histograms for each range

    # load results
    d = loadmat(path)

    bias = [d[0] for d in d['bias']]
    output = [d[0] for d in d['output']]
    target = [d[0] for d in d['target']]

    max_en = int(max(target)) + 0.5
    min_en = int(min(target)) - 0.5

    # Energy Transfomation for position data only
    # max_en_copy = max_en
    # min_en_copy = min_en
    # max_en = int(680.058 / min_en_copy)
    # min_en = int(680.058 / max_en_copy)

    if min_en >= 0.5:
        pass
    else:
        min_en = 0.5

    # Combined results array
    comb = np.stack([target, output, bias], axis=1)

    for i in np.arange(min_en, max_en, 1):
        if i != 7.5:  # For layer changing purposes only!
            continue
        bottom = i
        top = i + 1

        tmp_bias = []
        tmp_out = []
        tmp_tar = []

        for k, t in enumerate(target):

            #   Conversion for Position data only!
            # t = int(680.058 / t)

            if bottom <= t <= top:
                tmp_out.append(output[k])
                tmp_tar.append(target[k])
                tmp_bias.append(bias[k])

        E = bottom + 0.5
        theo_std = 0.2 * np.sqrt(E)
        mean, std = norm.fit(tmp_bias)

        assert len(tmp_bias) == len(tmp_tar) == len(tmp_out)

        fig, axs = plt.subplots(2, 2)

        ax0 = axs[0][0]
        ax0.hist(tmp_bias, bins=len(tmp_bias), color='b')
        ax0.axis('equal')
        ax0.set_title("Energy Bias hist")

        ax1 = axs[0][1]
        ax1.hist(tmp_tar, bins=len(tmp_tar), color='g')
        ax1.axis('equal')
        ax1.set_title("Energy TrueValue hist")

        ax1 = axs[1][0]
        ax1.hist(tmp_out, bins=len(tmp_out), color='r')
        ax1.axis('equal')
        ax1.set_title("Energy Predicted hist")

        # fig.suptitle(f'Energy range:{bottom} - {top} \n Theoretical std: {theo_std}, Bias std: {std} '
        #              f'\n {len(tmp_bias)} samples out of {len(bias)} samples')

        fig.suptitle(f'Energy range:{bottom} - {top} \n Bias std: {std}, percentage: {percentage}, up to layer: {layer_masking_lim}')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.savefig(fig_path / "LayerMasking" / f'Range_{bottom}-{top}_percent_{percentage}_{layer_masking_lim}.png')
        plt.savefig(fig_path / "LayerMasking" / f'range_{bottom}-{top}.png')
        plt.close(fig)


if __name__ == '__main__':
    # for i in range(1, 10, 2):
    #     load_predict_energies(fig_path / "LayerMasking" / f"Results_part_change{i / 10}_6.mat", i / 10, 6)
    # load_predict_energies(fig_path / "LayerMasking" / f"Results_part_change1_6.mat", 1, 6)
    load_predict_energies(fig_path / f"results.mat", 0, 0)


