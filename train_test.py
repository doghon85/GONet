import os, time, torch

import numpy as np
from tqdm import tqdm
import torch.optim as optim
from utils import perdictive_ability, make_com_order


def train(model, train_dataloader, save_path, device, lr, epoch, h2, type, make_com_t, k):
    save_model_path = save_path + 'models/'
    os.makedirs(save_model_path, exist_ok=True)

    model._init_weights()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    log = open(os.path.join(save_path, type+'_log.txt'), 'a')
    log.write(f'current {k}th fold\n')
    log.write(f'{type} training process\n')
    log.close()

    s_time = time.time()
    for ep in range(epoch):
        model.train()

        for i, data in enumerate(train_dataloader):
            X, true_y = data['X'].to(device), data['y'].to(device)
            pred_y, pred_order = model(X, train=True)
            
            true_com_order = make_com_order(true_y, make_com_t).to(device)
            trn_pa = perdictive_ability(true_y, pred_y, h2)
            loss1 = (true_y - pred_y).abs().mean()
            loss2 = criterion(pred_order, true_com_order)
            loss = loss1 + loss2
            
            monitor = f'epoch {ep + 1} ({i + 1}/{len(train_dataloader)}): loss ({round(float(loss), 5)}), trn p.a: ({round(trn_pa, 5)})'
            print(monitor + ' ' * 10, end='\r')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log = open(os.path.join(save_path, type + '_log.txt'), 'a')
        log.write(f'{monitor}\n')
        log.close()
        
    running_time = time.time() - s_time
    torch.save(model.state_dict(), os.path.join(save_model_path, f'last.pth'))
    del loss, optimizer

    return running_time

def test(model, test_dataloader, device, save_path, h2, type, k, ymean):
    model.eval()

    test_true_y = []
    test_pred_y = []
    
    s_time = time.time()
    for data in tqdm(test_dataloader):
        X = data['X'].to(device)
        a = data['a'].to(device)
        
        with torch.no_grad():
            batch_pred_y, _ = model(X, train=False)
            test_true_y.append(data['y'].to(device))
            test_pred_y.append(ymean+ (batch_pred_y-ymean + a)/2)
            
    running_time = time.time() - s_time

    test_true_y = torch.cat(test_true_y).to(device)
    test_pred_y = torch.cat(test_pred_y).to(device)
    
    pa = perdictive_ability(test_true_y, test_pred_y, h2)
    MAE = (test_true_y - test_pred_y).abs().mean().item()

    CI_y = test_true_y.mean()
    CI = pa / ( (MAE/CI_y) + 1 )
    print(f'current fold: {k}, Type: {type}, predictive ability: {pa}, MAE: {MAE}, CI: {CI}\n')

    with open(os.path.join(save_path, f'sol_avg_CI_{type}.txt'), 'a') as save:
        save.write(f'current fold: {k}\n')
        save.write(f'Type: {type}\n')
        save.write(f'predictive ability: {pa}\n')
        save.write(f'MAE: {MAE}\n')
        save.write(f'CI:\t{CI}\n')
        save.write(f'\n')
        
    return running_time
