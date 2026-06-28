from torch.utils.data import Dataset, DataLoader
import os

class PatchDataset:
    
    def __init__(self, root, split_path):
        self.root = root
        self.split_path = split_path 
        self.input_tir = os.path.join(root, split_path)