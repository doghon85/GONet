import math
import torch
from tqdm import tqdm


def mixed_model(Z, A, lamb, y):
        y = y - y.mean()
        e  = torch.inverse(Z.T @ Z + A *lamb) @ Z.T @ y
        return e

def call_Z(ref_len, whole_len):
    Z = torch.zeros((ref_len, whole_len),dtype=torch.float32)
    for i in range(ref_len): Z[i,i] = 1
    return Z

def make_G(X, invers=True, device='cpu'):

    n,k = X.shape
    pi = X.sum(0)/(2*n) # X.mean(0)/2
    P = (pi).unsqueeze(0) 

    # make A (dummay pedigree)
    A = torch.eye(len(X))

    # make G
    Z = X - 2*P 
    G = (Z @Z.T)  /(2*(pi*(1-pi)).sum()) 
    G = G * 0.99 + A * 0.01

    if invers:
        return torch.inverse(G)
    
    return G

def GBLUP(train_X, test_X, train_y, test_y, h2):

    X = torch.cat([train_X, test_X], dim=0)
    train_len = len(train_y)

    # Compute G
    M = X - 1
    pi = X.mean(0)/2
    P = 2*(pi-0.5)
    M = M - P 
    G = (M @M.T)/(2*(pi*(1-pi)).sum())

    G_inv = torch.inverse(G)
    Z = torch.zeros((train_len, len(G_inv)),dtype=torch.float32)
    for i in range(train_len): Z[i][i] = 1
    y_c = (train_y - train_y.mean()).unsqueeze(1)
    y_c = train_y.unsqueeze(1)
    lamb = h2/(1 - h2)
    
    y_gblup = torch.inverse(Z.T @ Z + G_inv*lamb) @ Z.T @ y_c

    return y_gblup[:,0][:train_len], y_gblup[:,0][train_len:], Z

def perdictive_ability(true, pred, h2=1):
    pred_mean = pred.mean()
    true_mean = true.mean()

    f1 = torch.sum((pred - pred_mean) * (true - true_mean))
    f2 = torch.sqrt(torch.sum((pred - pred_mean)**2) * torch.sum((true - true_mean)**2))
    if f2 == 0:
        return 0

    cor = f1/f2 
    return float(cor/math.sqrt(h2))

def make_com_order(true_y, t = 1):
    true_com = torch.zeros((true_y.shape[0]//2, 3))
    true_y = true_y.to(torch.int)
    res = true_com.shape[0]
    for i in range(true_com.shape[0]):
        if true_y[i] + t < true_y[i+res]: # <
            true_com[i][0] = 1
        elif true_y[i] > true_y[i+res] + t: # >
            true_com[i][2] = 1
        else: # ==
            true_com[i][1] = 1

    return true_com
