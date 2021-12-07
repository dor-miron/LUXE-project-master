from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, ConcatDataset
from data_loader import EcalDataIO
import torch
import random
from pathlib import Path
import numpy as np
from collections import Counter

CSV_LEN = 25410


# ------------------------------------ DATALOADERS ------------------------------------- #


class CE_Loader(BaseDataLoader):
    """
    Generates a DL from the existing files - concatenates the chunk_num of files.
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 chunk_low_num=0, chunk_high_num=1, partial_change=None, layer_change_lim=None):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])  # Not in use for now
        self.data_dir = Path(__file__).parent.parent / Path(data_dir)
        self.partial_change = partial_change

        dl = []
        for i in range(chunk_low_num, chunk_high_num):
            edep_file = self.data_dir / f"signal.al.elaser.edeplist{i}.mat"
            en_file = self.data_dir / f"signal.al.elaser.energy{i}.mat"
            # xy_file = self.data_dir / f"signal.al.elaser.trueXY{i}.mat"
            xy_file = self.data_dir / f"signal.al.elaser.energy{i}.mat"
            dataset = Continous_Energy_Data(edep_file, xy_file, status='train', energy=0, en_file=en_file,
                                            partial_change=partial_change, layer_change_lim=layer_change_lim)
            dl.append(dataset)

        self.dataset = ConcatDataset(dl)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class moment_loader(BaseDataLoader):
    """
    Generates a DL from the existing files - concatenates the chunk_num of files.
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 chunk_low_num=0, chunk_high_num=1, partial_change=None, layer_change_lim=None):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])  # Not in use for now

        if training == True:
            self.dataset = torch.load(Path(data_dir) / "train//train.pt")
        else:
            self.dataset = torch.load(Path(data_dir) / "test//test.pt")
        # self.dataset = torch.load(Path(data_dir) / "train//train.pt")

        print("Dataset len: ", len(self.dataset))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class rand_loader(BaseDataLoader):
    """
    Generates a DL from the existing files - concatenates the chunk_num of files.
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 chunk_low_num=0, chunk_high_num=1, partial_change=None, layer_change_lim=None):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])  # Not in use for now

        if training == True:
            self.dataset = torch.load(Path(data_dir) / "train//train.pt")
        else:
            self.dataset = torch.load(Path(data_dir) / "test//test.pt")

        # self.dataset = ConcatDataset([self.dataset, self.rand_ds])
        self.dataset = Random_DS(len(self.dataset))

        # self.dataset = torch.load(Path(data_dir) / "train//train.pt")
        print("Dataset len: ", len(self.dataset))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# __________________________________________DATASETS_______________________________________________________________
# Fitting for the new DS for continous energies
class Continous_Energy_Data(Dataset):

    def __init__(self, en_dep_file, xy_file, transform=None, status='train', energy=0, en_file=None,
                 partial_change=None, layer_change_lim=None):

        self.en_dep = EcalDataIO.ecalmatio(en_dep_file)  # Dict with 100000 samples {(Z,X,Y):energy_stamp}
        self.entry_dict = EcalDataIO.xymatio(xy_file)
        self.initial_energy = energy
        self.num_showers = 1
        self.energies = EcalDataIO.energymatio(en_file)
        self.partial_change = partial_change
        self.layer_change_lim = layer_change_lim

        # Eliminate multiple numbers of some kind
        # del_list = []
        # for key in self.energies:
        #     if 8 > len(self.energies[key]) > 4:
        #         del_list.append(key)
        # for d in del_list:
        #     del self.energies[d]
        #     del self.en_dep[d]
        #     del self.entry_dict[d]

    def __len__(self):
        return len(self.en_dep)
        # return 10

    # Randomly change values of sample to 0 - amount of num*(1-partial_change)
    def change_sample(self, sample: dict):
        indices = np.random.choice(np.arange(len(sample.keys())), replace=False,
                                   size=int(len(sample.keys()) * self.partial_change))
        for idx in indices:
            k = list(sample.keys())[idx]
            z, x, y = k
            if z < self.layer_change_lim:
                continue
            sample[k] = 0
        return sample

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        d_tens = torch.zeros((110, 11, 21))  # Formatted as [x_idx, y_idx, z_idx]

        key = list(self.en_dep.keys())[idx]

        tmp = self.en_dep[key]

        if self.partial_change != 1:  # 1 means No changing the data. partial_change < 1 - change this percentage of the data by the wanted function
            tmp = self.change_sample(tmp)

        # for z, x, y in tmp.keys():
        for z, x, y in tmp:
            d_tens[x, y, z] = tmp[(z, x, y)]

        entry = torch.Tensor(self.entry_dict[key])
        # true_xy = PositionConverter.PadPosition(entry[0].item(), entry[1].item())

        d_tens = d_tens.unsqueeze(0)  # Only in conv3d
        sample = (d_tens, entry, self.initial_energy)

        if self.energies:
            # sample = (d_tens, entry, self.energies[key][0])
            sample = (d_tens, sum(entry), sum(self.energies[key]) / len(self.energies[key]))
            # if sample[1].shape[0] == 4:
            #     print("hi")
            # print(sample[0].shape, sample[1].shape, sample[2].shape)

        return sample


class moment_energy_Data(Dataset):

    def __init__(self, en_dep_file, en_file, transform=None, status='train', moment=1, min_shower_num=0,
                 max_shower_num=10000):

        self.en_dep = EcalDataIO.ecalmatio(en_dep_file)  # Dict with 100000 samples {(Z,X,Y):energy_stamp}
        self.energies = EcalDataIO.energymatio(en_file)
        self.moment = moment

        # Eliminate multiple numbers of some kind
        if min_shower_num > 0:
            del_list = []
            for key in self.energies:
                if len(self.energies[key]) < min_shower_num or len(self.energies[key]) >= max_shower_num:
                    del_list.append(key)
            for d in del_list:
                del self.energies[d]
                del self.en_dep[d]

    def __len__(self):
        return len(self.en_dep)

    def calculate_moment(self, moment_num, en_list, normalize=True):
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
            l.append(val ** 2)
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
                    t = val ** i
                    l.append(t)

            tmp = torch.mean(torch.Tensor(l))
            res.append(tmp)

        return res

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        d_tens = torch.zeros((110, 11, 21))  # Formatted as [x_idx, y_idx, z_idx]

        key = list(self.en_dep.keys())[idx]

        tmp = self.en_dep[key]

        # for z, x, y in tmp.keys():
        for z, x, y in tmp:
            d_tens[x, y, z] = tmp[(z, x, y)]

        d_tens = d_tens.unsqueeze(0)  # Only in conv3d

        en_list = torch.Tensor(self.energies[key])

        num_showers = len(en_list)

        moment = self.calculate_moment(self.moment, en_list, True)
        # moment = self.calculate_moment(2, en_list)
        mean = moment[0]
        var = moment[1]
        third = moment[2]
        fano = var / mean

        # en_mean =torch.mean(en_list)
        # en_sum = torch.sum(en_list)

        # sample = (d_tens, mean, var, third, num_showers)
        sample = en_list

        return d_tens, torch.Tensor(moment), num_showers


class Bin_energy_data(Dataset):

    def __init__(self, en_dep_file, en_file, transform=None, status='train', moment=1, min_shower_num=0,
                 max_shower_num=10000, file=0):

        self.en_dep = EcalDataIO.ecalmatio(en_dep_file)  # Dict with 100000 samples {(Z,X,Y):energy_stamp}
        self.energies = EcalDataIO.energymatio(en_file)
        self.moment = moment
        self.file = file

        # Eliminate multiple numbers of some kind
        if min_shower_num > 0:
            del_list = []
            for key in self.energies:
                if len(self.energies[key]) < min_shower_num or len(self.energies[key]) >= max_shower_num:
                    del_list.append(key)
            for d in del_list:
                del self.energies[d]
                del self.en_dep[d]

    def __len__(self):
        return len(self.en_dep)

    def calculate_moment(self, moment_num, en_list, normalize=True):
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
            l.append(val ** 2)
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
                    t = val ** i
                    l.append(t)

            tmp = torch.mean(torch.Tensor(l))
            res.append(tmp)

        return res

    def random_sample_for_addition(self, data, n, num_samples):

        # en_dep = EcalDataIO.ecalmatio("C:\\Users\\elihu\\PycharmProjects\\LUXE\\LUXE-project-master\\data\\raw"
        #                               "\\signal.al.elaser.IP05.edeplist.mat")
        # energies = EcalDataIO.energymatio("C:\\Users\\elihu\\PycharmProjects\\LUXE\\LUXE-project-master\\data\\raw"
        #                                   "\\signal.al.elaser.IP05.energy.mat")

        samples = random.sample(list(self.en_dep.keys()), num_samples)
        # while True:
        #     if len(self.energies[samples[0]]) != 1:
        #         samples = random.sample(list(self.en_dep.keys()), num_samples)
        #     else:
        #         break

        sample = torch.zeros((110, 11, 21))  # Formatted as [x_idx, y_idx, z_idx]
        N = 0
        for key in samples:
            N += len(self.energies[key])
            tmp = self.en_dep[key]
            # sum the samples:
            for z, x, y in tmp:
                sample[x, y, z] = sample[x, y, z] + tmp[(z, x, y)]

        print(f"Orig - {n}, Add - {N}")
        data = data + sample
        n = n + N
        print(f"sum - {n}")
        return data, n

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        d_tens = torch.zeros((110, 11, 21))  # Formatted as [x_idx, y_idx, z_idx]
        key = list(self.en_dep.keys())[idx]

        tmp = self.en_dep[key]

        # for z, x, y in tmp.keys():
        for z, x, y in tmp:
            d_tens[x, y, z] = tmp[(z, x, y)]


        ## ONLY 2 LAYER ON Y AXIS TRAINING
        # d_tens = d_tens[:, 4:7, 0:10]
        # d_tens = torch.transpose(d_tens, 0, 1)
        #########################################

        ############ Layer Removal Experiment  ############
        # Zerofi Z layers
        # for i in range(0, 7):
        #     d_tens[:, :, (20 - i)] = 0

        # Zerofi Y layers
        # for i in range(0, 6):
        #     d_tens[:, (10 - i), :] = 0
        #     d_tens[:, i, :] = 0
        ###################################

        ########## Alpha Experiment #########
        # alpha = 1
        #
        # d_tens = np.cos(np.deg2rad(alpha)) * d_tens + np.sin(np.deg2rad(alpha)) * torch.rand(torch.Size([110, 11, 21]))
        # d_tens = np.cos(np.deg2rad(alpha)) * d_tens + np.sin(np.deg2rad(alpha)) * torch.rand(torch.Size([110, 3, 10]))
        # d_tens = (1-alpha) * d_tens + (alpha) * torch.rand(torch.Size([110, 11, 21]))

        ####################


        ######### Normalization #############
        # if self.file == 3:
        #     d_tens = (d_tens - 0.0935) / 1.4025
        #########################################

        en_list = torch.Tensor(self.energies[key])

        num_showers = len(en_list)
        # Addition of samples for superposition test
        d_tens, num_showers = self.random_sample_for_addition(d_tens,  num_showers, 1)
        d_tens = d_tens.unsqueeze(0)  # Only in conv3d


        # final_list = [0] * 10
        final_list = [0] * 20
        bin_list = np.linspace(0, 13, 20)
        bin_list = np.linspace(0, 13, 10)
        binplace = np.digitize(en_list, bin_list)
        bin_partition = Counter(binplace)
        for k in bin_partition.keys():
            final_list[int(k) - 1] = bin_partition[k]
        n = sum(final_list)
        final_list = [f / n for f in final_list]
        # final_list = [f/100000 for f in final_list]


        return d_tens, final_list, num_showers, idx


class Random_DS(Dataset):
    # Generate random samples
    def __init__(self, len):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        bins = torch.zeros(20)
        bins[0] = 1
        # return torch.rand(torch.Size([1, 110, 11, 21])), bins, 0
        # return torch.ones(torch.Size([1, 110, 11, 21])), bins, 0
        # rand_bins = torch.FloatTensor(20).uniform_(50, 500)
        rand_bins = torch.ones(20)
        return torch.rand(torch.Size([1, 110, 11, 21])), rand_bins, torch.sum(rand_bins), 0.
