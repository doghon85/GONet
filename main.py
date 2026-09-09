# deepGBLUP	0.713	0.704	0.743	0.601
import os, time, torch
from torch.utils.data import DataLoader

from model import GONet
from dataset import load_dataset
from train_test import train, test


kfold = range(1, 6, 1)
for target in ['PFAI', 'MS']:
    for k in kfold:
        # data path
        raw_path = '../../data/geno.txt'
        phen_path = '../../data/pheno.txt' 
        train_path = f'../../data/5fold/train{k}.txt'
        test_path = f'../../data/5fold/test{k}.txt'
    
        # train cofig
        config = {
            'lr': 1e-4,
            'epoch': 150,
            'batch_size': 64,
            'make_com_t': 3,
            'target': target,
            'h2': {'PFAI': 0.262110, 'MS': 0.244917},
        }
        if config['target'] == 'PFAI':
            config['make_com_t'] = 0.1
        if config['target'] == 'MS':
            config['make_com_t'] = 0

        training = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        save_path = f'result_1_{config["target"]}_t{config["make_com_t"]}_lr{config["lr"]}/{k}/'  

        os.makedirs(save_path, exist_ok=True)
        s_time = time.time()

        # Load Dataset
        train_dataset, test_dataset, num_snp, ymean = load_dataset(raw_path, phen_path, train_path, test_path, config['h2'][config['target']], config['target'], device)
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Load model
        if training:
            lr = config['lr']
            print('\nk :', k)
            print(f'start model_{config["target"]} training')
            model = GONet(ymean, num_snp, device).to(device)
            train_time = train(
                model, train_dataloader, save_path, device, lr, config['epoch'], 
                config['h2'][config['target']], config['target'], config['make_com_t'], k
                )

        # Test model
        model = GONet(ymean, num_snp, device).to(device)
        model.load_state_dict(torch.load(save_path+'models/last.pth', map_location=device))
        test_time = test(
                model,
                test_dataloader,
                device, save_path, config['h2'][config['target']], config['target'], k, ymean
                )

        if training:
            with open(os.path.join(save_path,'setting.txt'),'a') as save:
                save.write(f'kfold : {k}\n')
                save.write(f'Path of raw file: {raw_path}\n')
                save.write(f'Path of phenotype file: {phen_path}\n')
                save.write('-'*50+'\n')
                for key in config:
                    save.write("{0} : {1}\n".format(key, config[key]))
                save.write(f'Device: {device}\n')
                save.write('-'*50+'\n')
                save.write(f'{config["target"]} ::training time : {train_time}\n')
