import random
import librosa
import numpy as np
from torch.utils.data import Dataset
from TF_MASK import *


def load_audio(feature, mode='train', spec_len=192):
    if mode == 'train':
        crop_start = random.randint(0, feature.shape[2] - spec_len)
        spec_mag = feature[:,:, crop_start:crop_start + spec_len]
    else:
        spec_mag = feature[:,:, :192]  

    mean = np.mean(spec_mag, axis=1, keepdims=True)
    std = np.std(spec_mag, axis=1, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    return spec_mag

class CustomDataset(Dataset):
    def __init__(self, data_path, model='train'): 
        super(CustomDataset, self).__init__()
        self.dataset = np.load(data_path, allow_pickle=True)
        self.model = model

    def __getitem__(self, idx):
        feature, label = self.dataset[idx]
        feature = load_audio(feature, mode=self.model)
        return feature, label

    def __len__(self):
        return len(self.dataset)
