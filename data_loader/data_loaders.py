from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, ConcatDataset
from data_loader import EcalDataIO
import torch
from pathlib import Path
import numpy as np


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


CSV_LEN = 25410


# ------------------------------------ CONTEN DATASET DEFINITION ------------------------------------- #


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
        self.data_dir = Path(data_dir)
        self.partial_change = partial_change

        dl = []
        for i in range(chunk_low_num, chunk_high_num):
            edep_file = self.data_dir / f"signal.al.elaser.edeplist{i}.mat"
            en_file = self.data_dir / f"signal.al.elaser.energy{i}.mat"
            xy_file = self.data_dir / f"signal.al.elaser.trueXY{i}.mat"
            dataset = Continous_Energy_Data(edep_file, xy_file, status='train', energy=0, en_file=en_file,
                                            partial_change=partial_change, layer_change_lim =layer_change_lim)
            dl.append(dataset)

        self.dataset = ConcatDataset(dl)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# Fitting for the new DS for continout energies
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

        if self.partial_change != 1:    # 1 means No changing the data. partial_change < 1 - change this percentage of the data by the wanted function
            tmp = self.change_sample(tmp)

        # for z, x, y in tmp.keys():
        for z, x, y in tmp:
            d_tens[x, y, z] = tmp[(z, x, y)]

        entry = torch.Tensor(self.entry_dict[key])
        # true_xy = PositionConverter.PadPosition(entry[0].item(), entry[1].item())

        d_tens = d_tens.unsqueeze(0)  # Only in conv3d
        sample = (d_tens, entry, self.initial_energy)

        if self.energies:
            sample = (d_tens, entry, self.energies[key][0])

        return sample
