import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import glob
import os
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import CASIADataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import tqdm

from datasets.transforms import RandomCrop, Resize, ToTensor

def load_model(args):
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/../checkpoints/{args.group}_{args.name}/*.pt'
    checkpoints = glob.glob(checkpoint_path)
    sorted_checkpoints = sorted(checkpoints, key = lambda x: int(x.split('_')[-1][:-3]))

    for checkpoint in sorted_checkpoints[:-1]:
        # os.unlink(checkpoint)
        pass

    return torch.load(sorted_checkpoints[-1])

def load_model_by_name(name):
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/../checkpoints/{name}/*.pt'
    checkpoints = glob.glob(checkpoint_path)
    sorted_checkpoints = sorted(checkpoints, key = lambda x: int(x.split('_')[-1][:-3]))

    for checkpoint in sorted_checkpoints[:-1]:
        # os.unlink(checkpoint)
        pass

    return torch.load(sorted_checkpoints[-1])
