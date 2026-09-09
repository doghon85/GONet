import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import perdictive_ability, make_G, mixed_model, call_Z

class SNPDataset(Dataset):
    def __init__(self, X, a, ids, y):
        self.X = X
        self.y = y
        self.ids = ids
        self.a = a

    def __len__(self):
        return len(self.X)


    def __getitem__(self,idx):
        return {
            'X': self.X[idx], 'y': self.y[idx], 
            'id': self.ids[idx], 'a': self.a[idx]
        }

def load_dataset(raw_path, phen_path, train_path, test_path, h2, target, device='cpu'):
    random.seed(7)
    with open(phen_path) as phen_file:
        header = next(phen_file).split()

        phen = {}
        for line in phen_file:
            line_ = line.split()
            if target == 'PFAI':
                phen[line_[0]] = float(line_[-2])
            elif target == 'MS':
                phen[line_[0]] = float(line_[-1])


    # extract the train ids
    train_ids = []
    with open(train_path) as train_ids_file:
        for line in train_ids_file:
            train_ids.append(line[:-1])

    test_ids = []
    with open(test_path) as test_ids_file:
        for line in test_ids_file:
            test_ids.append(line[:-1])

    # Load snp and Rearrange phen data 
    raw_file = open(raw_path)
    header = next(raw_file).split()

    train_X = []
    train_y = []
    test_X = []
    test_y = []

    for line in tqdm(raw_file):
        line_ = line.split()
        
        if line_[0] in train_ids:
            train_X.append(line_[1:])
            train_y.append(phen[line_[0]])
        elif line_[0] in test_ids:
            test_X.append(line_[1:])
            test_y.append(phen[line_[0]])
    raw_file.close() 
    
    print(len(train_X), len(train_y), len(test_X), len(test_y))
    
    if len(train_X) % 2 == 1:
        train_X = train_X[:-1]
        train_y = train_y[:-1]
        train_ids = train_ids[:-1]
        
    if len(test_X) % 2 == 1:
        test_X = test_X[:-1]
        test_y = test_y[:-1]
        
    train_X = torch.from_numpy(np.array(train_X, dtype=np.float32))
    train_y = torch.tensor(train_y, dtype=torch.float32)
    train_ids = np.array(train_ids, dtype=str)
    
    test_X = torch.from_numpy(np.array(test_X, dtype=np.float32))
    test_y = torch.tensor(test_y, dtype=torch.float32)
    test_ids = np.array(test_ids, dtype=str)
    
    X = torch.cat([train_X, test_X], dim=0)
    y = train_y.clone()
    Gi = make_G(X, invers=True, device=device)
    Z = call_Z(len(train_X), len(train_X)+len(test_X))
    glamb = (1 - h2)/h2
    
    a = mixed_model(Z, Gi, glamb, y)
    train_a, test_a = a[:len(train_X)], a[len(train_X):]
    
    train_dataset = SNPDataset(train_X, train_a, train_ids, train_y)
    test_dataset = SNPDataset(test_X, test_a, test_ids, test_y)
    return train_dataset, test_dataset, X.shape[1], y.mean()
