import fnmatch
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.optimize import curve_fit

from data_loader.data_loaders import Bin_energy_data

project_path = Path(__file__).parent.parent
fig_path = project_path / "saved\\figs\\"
res_path = project_path / "saved\\diff_run_res\\"


def evaluate_test(output, target, incdices, shower_nums):
    """
        This function evaluates the test results. The first part relates to the N prediction - assuming we have 1 class.
        The second part handles the 20 bin prediction.
    """
    file_tag = "x_pred"

    # evaluate_xy(output=output[:, 0], target=target[:, 0], run_num=file_tag)
    # np.savetxt(res_path / f'output_{file_tag}.txt', output[:, 0], delimiter="\n", fmt='%1.2f')


    # For each sample evaluate N and compare with Ntrue - get Nbias. Produce halina graphs with them and our graphs:
    # N_pred = np.sum(output, axis=1)  # Total number of showers in the test set
    # H_graphs(N_true=shower_nums, N_pred=N_pred, run_num=file_tag)  # Generate relevant graph images

    # Bins - sum over output bins and target bins. Compare graphs. produce PNG. Save the bins as text to calculate
    test_bins(output, target, shower_nums, bin_num=20, run_num=file_tag)

    ################################################################
    ######### Save idx list as txt file with the file tag ##########
    # np.savetxt(res_path / f'idx_run_{file_tag}.csv', [int(i) for i in incdices], delimiter=",", fmt='%i')
    ################################################################

    return


###################################
### Function for curve fitting ####
def normal(x, A, mu, sigma):
    return A * np.exp((-(x - mu) ** 2) / (2 * (sigma ** 2)))


###################################

def H_graphs(N_true, N_pred, run_num=0):
    """
    Generates the following graphs:
        - BiasN\\N historgram and normal dist
        - BiasN\\N vs N
        - BiasN\\N hist2d
        - N_pred vs N

    """
    N_bias_x = (N_true - N_pred) / N_true
    N_bias_y = (N_pred - N_true) / N_true

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    # fig.suptitle(f'') # Fig title
    ax0 = axs[0][0]
    ax1 = axs[0][1]
    ax2 = axs[1][0]
    ax3 = axs[1][1]

    # - BiasN\N historgram and normal dist
    # Fit a normal distribution to
    # the data:
    # mean and standard deviation
    bin_heights, bin_borders, _ = ax0.hist(N_bias_x, bins=len(N_bias_x), density=True, alpha=0.6,
                                           color='black', histtype=u'step')     # Plot the histogram.

    bin_heights[0] = 0
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    params, _ = curve_fit(normal, bin_centers, bin_heights, p0=[1., 0., 1.], maxfev=5000)   # Comment out for no fitting
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)    # Comment out for no fitting
    p = normal(x_interval_for_fit, params[0], params[1], params[2])     # Comment out for no fitting
    ax0.plot(x_interval_for_fit, p, 'k', linewidth=1, color='green')    # Comment out for no fitting

    """
    Sometimes when the results are bad - the fit does not converge and throws and error. In this case comment out 
    The marked lines
    """

    # title = f"Fit Values: {[f'{np.abs(p):2f}' for p in params]}"  # all of the fitted params
    title = f"Relative error prob. density. \n Fitted sigma {np.abs(params[2]):2f}"     # Comment out for no fitting
    ax0.set_title(title)    # Comment out for no fitting
    ax0.legend()
    ax0.set_ylabel('Probablity Density')
    ax0.set_xlabel('(N_true - N_reco) / N_true')

    # - BiasN\N hist2d
    gridx = np.linspace(min(N_true), max(N_true), 150)
    gridy = np.linspace(min(N_bias_y), max(N_bias_y), 150)
    h = ax1.hist2d(N_true, N_bias_y, bins=[gridx, gridy], cmin=1)
    fig.colorbar(h[3], ax=ax1)
    title = f"Relative Error vs N 2d histogram"
    ax1.set_title(title)
    ax1.set_ylim([min(N_bias_y), -min(N_bias_y)])
    ax1.set_ylabel('(N_reco - N_true) / N_true')
    ax1.set_xlabel('N_true')
    ax1.set_facecolor('white')

    # - BiasN\N vs N
    m = np.mean(N_bias_y)
    ax2.scatter(N_true, N_bias_y, marker='.', color='black')
    ax2.set_title(f'Bias/N vs N, mean : {m:2f}')
    ax2.set_ylabel('(N_reco - N_true) / N_true')
    ax2.set_xlabel('N_true')

    # - N_pred vs N
    ax3.scatter(N_true, N_pred, marker='.', color='black')
    ax3.set_title('N_pred vs N')
    ax3.set_ylabel('N_pred')
    ax3.set_xlabel('N_true')

    ##############################################################################
    # Save just the portion _inside_ the second axis's boundaries - separate graph saves.
    # extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig(res_path / f'ax0_{run_num}.png', bbox_inches=extent.expanded(1.5, 1.5))
    #
    # extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig(res_path / f'ax1_{run_num}.png', bbox_inches=extent.expanded(1.5, 1.7))
    ##############################################################################

    plt.tight_layout()
    plt.savefig(res_path / f'Nresults_run_{run_num}.png')
    plt.clf()
    return


def evaluate_xy(output, target, run_num='0'):
    """
    Generates the following graphs:
        - BiasX\\X historgram and normal dist
    """
    N_bias_x = (output - target) / target

    fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    # fig.suptitle(f'') # Fig title
    ax0 = axs


    # - BiasN\N historgram and normal dist
    # Fit a normal distribution to
    # the data:
    # mean and standard deviation
    bin_heights, bin_borders, _ = ax0.hist(N_bias_x, bins=100, density=True, alpha=0.6,
                                           color='black', histtype=u'step')  # Plot the histogram.

    bin_heights[0] = 0
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    params, _ = curve_fit(normal, bin_centers, bin_heights, p0=[1., 0., 1.],
                          maxfev=5000)  # Comment out for no fitting
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)  # Comment out for no fitting
    p = normal(x_interval_for_fit, params[0], params[1], params[2])  # Comment out for no fitting
    ax0.plot(x_interval_for_fit, p, 'k', linewidth=1, color='green')  # Comment out for no fitting

    """
    Sometimes when the results are bad - the fit does not converge and throws and error. In this case comment out 
    The marked lines
    """

    # title = f"Fit Values: {[f'{np.abs(p):2f}' for p in params]}"  # all of the fitted params
    title = f"Relative error prob. density. \n Fitted sigma {np.abs(params[2]):2f}"  # Comment out for no fitting
    ax0.set_title(title)  # Comment out for no fitting
    ax0.legend()
    ax0.set_ylabel('Probablity Density')
    ax0.set_xlabel('(X_true - X_reco) / X_true')

    plt.tight_layout()
    plt.savefig(res_path / f'X_results_{run_num}.png')
    plt.clf()

    return

def test_bins(output, target, nums, bin_num=10, name=None, run_num='0'):
    """Analysis of the bin results from test"""

    # Generate the bins
    total_out = [0] * bin_num
    total_target = [0] * bin_num
    bars = np.linspace(0, 13, bin_num)

    # Sum over al of the bin results for each bin.
    for i in range(bin_num):
        out_sum = sum(t[i] for t in output)
        target_sum = sum(t[i] for t in target)
        total_out[i] = out_sum
        total_target[i] = target_sum

    # Calculate the entropy of the bin lists for output and the truelabel bins.

    out_entropy = -sum([f * np.log(f) if f > 0 else 0 for f in total_out])
    tar_entropy = -sum([f * np.log(f) if f > 0 else 0 for f in total_target])
    b = out_entropy - tar_entropy

    print(f'Entopies: True distribution: {tar_entropy:.3f}, Predicted distribution: {out_entropy:.3f}, Bias: {b:.3f}')

    KL_1 = sum(
        [f * np.log((f + 0.0000001) / (total_target[i] + 0.0000001)) for i, f in enumerate(total_out)])
    KL_2 = sum(
        [f * np.log((f + 0.0000001) / (total_out[i] + 0.0000001)) for i, f in enumerate(total_target)])
    D = KL_1 + KL_2

    print(f'KL dist - KL1(q=target): {KL_1:.3f}, KL2(q=output): {KL_2:.3f}, Symmetric: {D:.3f}')
    print(f'total out: {["%.3f" % item for item in total_out]}')
    print(f'total target: {["%.3f" % item for item in total_target]}')
    print(f'total N: {sum(total_out)}, target N: {sum(total_target)}')
    bars = [float(f'{i:.2f}') for i in bars]

    # Text generation for bin legend
    text = 'Bin Energy range [GeV]: \n'
    for i in range(bin_num - 1):
        text += f'{i}: {bars[i]:.1f} - {bars[i + 1]:.1f} \n'
    # print(text)

    tot_mean_en_pred = []
    tot_mean_en_true = []
    for i in range(bin_num - 1):
        me = bars[i] + 0.35
        e_p = me * total_out[i]
        e_t = me * total_target[i]
        tot_mean_en_pred.append(e_p)
        tot_mean_en_true.append(e_t)

    mean_en_p = sum(tot_mean_en_pred) / sum(total_out)
    mean_en_t = sum(tot_mean_en_true) / sum(total_target)

    print(f'Mean E: {mean_en_p}, target E: {mean_en_t}')

    rng = [i + 1 for i in range(20)]
    plt.bar(rng, total_out, label='output', alpha=0.5)
    plt.errorbar(rng, total_out, yerr=(1 / np.sqrt(np.abs(total_out))), fmt="+", color="b")
    plt.bar(rng, total_target, label='true_val', alpha=0.3)

    plt.text(15.5, 0.015, text,
             bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3}, fontsize='x-small')

    plt.xticks(rng, rotation=65)
    plt.title(f'{len(output)} samples')
    plt.legend()
    plt.savefig(res_path / f'binsgraph_run_{run_num}.png')
    # plt.show()
    plt.clf()
    save = {'output bins': total_out, 'output entropy': out_entropy,
            'target bins': total_target, 'target entropy': tar_entropy,
            'KL': KL_2}

    res_file = open(res_path / f'bin_results_run_{run_num}.txt', 'wt')
    res_file.write(str(save))
    res_file.close()
    return


def calculate_moment_list(moment_num, en_list, normalize=True):
    """Calculate the n'th moment (up to moment_num) of a given energies list. Same function as in the dataset."""
    res = []
    if not torch.is_tensor(en_list):
        en_list = torch.Tensor(en_list)

    first = torch.mean(en_list)
    res.append(torch.mean(en_list))
    if moment_num == 1:
        return res

    l = []
    for val in en_list:
        # l.append((val - first) ** 2)
        l.append((val) ** 2)
    second = torch.mean(torch.Tensor(l))
    res.append(second)

    if moment_num == 2:
        return res

    for i in range(3, moment_num + 1):
        l = []
        for val in en_list:
            if normalize:
                # t = (val - first) ** i
                t = (val) ** i
                s = second ** i
                r = t / s
                l.append(r)
            else:
                # t = (val - first) ** i
                t = (val) ** i
                l.append(t)

        tmp = torch.mean(torch.Tensor(l))
        res.append(tmp)

    return res


def EDA(path):

    """
    This is a general function for exploring the data. The begining filters out the relevant file to look at, and
    then there are different tests and checks we did on the data.
    I choose the energy file, and then load the edep-lists if needed.
    """
    # Go through energy files and print histogram of positron amounts
    low = []
    mom = 10
    nrm = True
    total_en_1 = []
    total_en_2 = []
    total_en_3 = []
    f_1 = 0
    f_2 = 0
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, '*IP05.energy*'):
            final = []
            f_path = Path(path) / file
            mat = loadmat(f_path)

            # Load edep matrix.
            ff_path = Path(str(f_path)[:-10] + 'edeplist.mat')
            # edep_mat = loadmat(ff_path)

            del mat['__globals__']
            del mat['__header__']
            del mat['__version__']
            print(f'{str(file)}, len: {len(mat.keys())}')
            # del edep_mat['__globals__']
            # del edep_mat['__header__']
            # del edep_mat['__version__']
            i = 0
            nums_1 = []
            nums_2 = []
            nums_3 = []
            for key in mat.keys():
                num_p = mat[key].shape[0]

                if 9 < num_p < 610:
                    nums_1.append(num_p)
                    energies = torch.Tensor(mat[key][:, 0])
                    i += 1
                    total_en_1.extend(energies)

                # if 609 < num_p < 1219:
                #     nums_2.append(num_p)
                #     energies = torch.Tensor(mat[key][:, 0])
                #     i += 1
                #     total_en_2.extend(energies)
                #
                # if 1219 < num_p:
                #     nums_3.append(num_p)
                #     energies = torch.Tensor(mat[key][:, 0])
                #     i+=1
                #     total_en_3.extend(energies)

                # moments = calculate_moment_list(mom, energies, normalize=nrm)
                # print(f"num: {num_p}")
                # print("moments: ", moments)
                # moments.insert(0, num_p)
                # final.append((moments, num_p))

            final_list = [0] * 20
            bin_list = np.linspace(0, 13, 20)
            # bin_list = np.linspace(0, 13, 10)
            binplace = np.digitize(total_en_1, bin_list)
            bin_partition = Counter(binplace)

            for k in bin_partition.keys():
                final_list[int(k) - 1] = bin_partition[k]

            final_list = [f + 0.0000001 for f in final_list]

            n = sum(final_list)
            final_list = [f / n for f in final_list]
            ent = -(final_list * np.log(np.abs(final_list))).sum()

            f_1 = np.asarray(final_list, dtype=np.float64)

            rng = [i + 1 for i in range(20)]
            plt.bar(rng, final_list, label=f'output05_{min(nums_1)}-{max(nums_1)}_entropy-{ent:.1f}', alpha=0.5)
            continue

        if fnmatch.fnmatch(file, '*IP03.energy*'):
            final = []
            f_path = Path(path) / file
            mat = loadmat(f_path)
            ff_path = Path(str(f_path)[:-10] + 'edeplist.mat')
            # edep_mat = loadmat(ff_path)

            del mat['__globals__']
            del mat['__header__']
            del mat['__version__']
            print(f'{str(file)}, len: {len(mat.keys())}')
            # del edep_mat['__globals__']
            # del edep_mat['__header__']
            # del edep_mat['__version__']
            i = 0
            nums_1 = []
            nums_2 = []
            nums_3 = []
            for key in mat.keys():
                num_p = mat[key].shape[0]

                if 9 < num_p < 610:
                    nums_1.append(num_p)
                    energies = torch.Tensor(mat[key][:, 0])
                    i += 1
                    total_en_1.extend(energies)

                # if 609 < num_p < 1219:
                #     nums_2.append(num_p)
                #     energies = torch.Tensor(mat[key][:, 0])
                #     i += 1
                #     total_en_2.extend(energies)
                #
                # if 1219 < num_p:
                #     nums_3.append(num_p)
                #     energies = torch.Tensor(mat[key][:, 0])
                #     i+=1
                #     total_en_3.extend(energies)

                # moments = calculate_moment_list(mom, energies, normalize=nrm)
                # print(f"num: {num_p}")
                # print("moments: ", moments)
                # moments.insert(0, num_p)
                # final.append((moments, num_p))

            final_list = [0] * 20
            bin_list = np.linspace(0, 13, 20)
            # bin_list = np.linspace(0, 13, 10)
            binplace = np.digitize(total_en_1, bin_list)
            bin_partition = Counter(binplace)

            for k in bin_partition.keys():
                final_list[int(k) - 1] = bin_partition[k]

            final_list = [f + 0.0000001 for f in final_list]

            n = sum(final_list)
            final_list = [f / n for f in final_list]
            ent = -(final_list * np.log(np.abs(final_list))).sum()

            f_2 = np.asarray(final_list, dtype=np.float64)

            rng = [i + 1 for i in range(20)]
            plt.bar(rng, final_list, label=f'output03_{min(nums_1)}-{max(nums_1)}_entropy-{ent:.1f}', alpha=0.5)
            # plt.errorbar(rng, final_list, yerr=(1 / np.sqrt(np.abs(final_list))), fmt="+", color="b")
            # plt.bar(rng, final_list, label='true_val', alpha=0.3)
            continue

            # final_list = [0] * 20
            # bin_list = np.linspace(0, 13, 20)
            # # bin_list = np.linspace(0, 13, 10)
            # binplace = np.digitize(total_en_2, bin_list)
            # bin_partition = Counter(binplace)
            #
            # for k in bin_partition.keys():
            #     final_list[int(k) - 1] = bin_partition[k]
            # final_list = [f + 0.0000001 for f in final_list] + f_1

            # n = sum(final_list)
            # final_list = [f / n for f in final_list]
            # ent = -(final_list * np.log(np.abs(final_list))).sum()

            # f_2 = np.asarray(final_list, dtype=np.float64)
            #
            # rng = [i + 1 for i in range(20)]
            # plt.bar(rng, final_list, label=f'output(1+2)_{min(nums_1)}-{max(nums_2)}_entropy-{ent:.1f}', alpha=0.5)
            # # plt.errorbar(rng, final_list, yerr=(1 / np.sqrt(np.abs(final_list))), fmt="+", color="b")
            # # plt.bar(rng, final_list, label='true_val', alpha=0.3)
            #
            #
            # final_list = [0] * 20
            # bin_list = np.linspace(0, 13, 20)
            # # bin_list = np.linspace(0, 13, 10)
            # binplace = np.digitize(total_en_3, bin_list)
            # bin_partition = Counter(binplace)
            #
            # for k in bin_partition.keys():
            #     final_list[int(k) - 1] = bin_partition[k]
            # final_list = [f + 0.0000001 for f in final_list] + f_1 + f_2

            # n = sum(final_list)
            # final_list = [f / n for f in final_list]
            # ent = -(final_list * np.log(np.abs(final_list))).sum()
            #
            # f_3 = np.asarray(final_list, dtype=np.float64)
            #
            # rng = [i + 1 for i in range(20)]
            # plt.bar(rng, final_list, label=f'output(1+2+3)_{min(nums_1)}-{max(nums_3)}_entropy-{ent:.1f}', alpha=0.5)
            # plt.errorbar(rng, final_list, yerr=(1 / np.sqrt(np.abs(final_list))), fmt="+", color="b")
            # plt.bar(rng, final_list, label='true_val', alpha=0.3)

            # KL DIST
        else:
            continue

    dist_1_2 = np.sum(np.where(f_1 != 0, f_1 * np.log(f_1 / f_2), 0))
    # dist_1_3 = np.sum(np.where(f_1 != 0, f_1 * np.log(f_1 / f_3), 0))
    # dist_2_3 = np.sum(np.where(f_2 != 0, f_2 * np.log(f_2 / f_3), 0))

    print(f'dists: 1-2: {dist_1_2}')

    plt.xticks(rng, rotation=65)
    plt.title(f'{i} samples of 03 and 05 \n dists: 1-2: {dist_1_2:.4f}')
    plt.legend()
    plt.savefig(res_path / f'{i} samples 03-05 normalized.png')
    plt.show()
    # plt.clf()

    # save = {'output bins': final_list}
    #
    # res_file = open(res_path / f'03analysis-3.txt', 'wt')
    # res_file.write(str(save))
    # res_file.close()

    '''
            df = pd.DataFrame()
            df['num_showers'] = [x[1] for x in final]

            for i in range(mom):
                cur_moments = [x[0][i] for x in final]
                cur_nums = [x[1] for x in final]

                df[f'{i + 1}_moment'] = [float(m) for m in cur_moments]

                # if i == 0:
                #     plt.scatter(cur_nums, cur_moments, label=f'{i + 1}st moment', alpha=0.3)
                # elif i == 1:
                #     plt.scatter(cur_nums, cur_moments, label=f'{i + 1}nd moment', alpha=0.3)
                # elif i == 2:
                #     plt.scatter(cur_nums, cur_moments, label=f'{i + 1}rd moment', alpha=0.3)
                # else:
                #     plt.scatter(cur_nums, cur_moments, label=f'{i + 1}th moment', alpha=0.5)

            # plt.title(f"moments over num shower, normalization: {norm}")
            # plt.legend()
            # plt.show()

            writer = pd.ExcelWriter(f'Moments_normalized{str(file)[17:21]}.xlsx', engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Moments', startrow=1, header=False, index=False)
            workbook = writer.book
            worksheet = writer.sheets['Moments']
            (max_row, max_col) = df.shape
            column_settings = [{'header': column} for column in df.columns]
            worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})
            worksheet.set_column(0, max_col - 1, 12)
            writer.save()

            del df
    '''


def merge_and_split_data(path, relation, moment, min_shower_num, max_shower_num, file):
    """
    Merge the raw data files into concatenated dataset then split by the relation to train and test set and save in the
     folders.
    Notice that the train is split into train\valid sets in the train function later.
    """
    dl = []
    for i in file:
        edep_file = path / "raw" / f"signal.al.elaser.IP0{i}.edeplist.mat"
        en_file = path / "raw" / f"signal.al.elaser.IP0{i}.energy.mat"

        dataset = Bin_energy_data(edep_file, en_file, moment=moment,
                                  min_shower_num=min_shower_num, max_shower_num=max_shower_num, file=i)
        dl.append(dataset)

    dataset = torch.utils.data.ConcatDataset(dl)

    # mean, std = get_mean_and_std(dataset)
    # print(mean, std)

    train_d, test_d = torch.utils.data.random_split(dataset, [int(relation * len(dataset)) + 1,
                                                              int((1 - relation) * len(dataset))])

    torch.save(train_d, path / "train//train.pt")
    torch.save(test_d, path / "test//test.pt")


def get_mean_and_std(dataloader):
    """Get the mean and variance values of the dataset - meaning the mean and variance of each matrix sample"""
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _, _ in dataloader:
        channels_sum += torch.mean(data)
        channels_squared_sum += torch.mean(data ** 2)
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


if __name__ == '__main__':
    # EDA("C:\\Users\\elihu\\PycharmProjects\\LUXE\\nongitdata\\Multiple Energies\\")

    data_path = Path("C:\\Users\\elihu\\PycharmProjects\\LUXE\\LUXE-project-master\\data\\")
    merge_and_split_data(data_path, 0.8, moment=3, min_shower_num=1, max_shower_num=50000, file=[5])
    exit()
