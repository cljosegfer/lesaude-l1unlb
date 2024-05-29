
import torch
import torch.nn as nn
import os
import numpy as np

from tqdm import tqdm

from utils import plot_log, export, get_inputs, plot_epoch
from hparams import BATCH_SIZE, NUM_WORKERS

class Runner():
    def __init__(self, device, model, database, split, model_label = 'baseline'):
        self.device = device
        self.model = model
        self.database = database
        self.model_label = model_label
        if not os.path.exists('output/{}'.format(model_label)):
            os.makedirs('output/{}'.format(model_label))
            os.makedirs('output/{}/fig'.format(model_label))
        
        self.trn_ds = split(database, database.pretrn_idx_dict)
        self.val_ds = split(database, database.val_idx_dict)
    
    def train(self, epochs):
        self.model = self.model.to(self.device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = 3e-4)

        trn_dl = torch.utils.data.DataLoader(self.trn_ds, batch_size = BATCH_SIZE, 
                                             shuffle = True, num_workers = NUM_WORKERS)
        val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size = BATCH_SIZE, 
                                             shuffle = False, num_workers = NUM_WORKERS)

        log = []
        minimo = 1e6
        # best = self.model
        for epoch in range(epochs):
            print('-- epoch {}'.format(epoch))
            trn_log = self._train_loop(trn_dl, optimizer, criterion)
            val_log = self._eval_loop(val_dl, criterion)
            plot_epoch(self.model_label, trn_log, val_log, epoch)
            log.append([sum(trn_log) / len(trn_dl), val_log])
            plot_log(self.model_label, log)
            if val_log < minimo:
                minimo = val_log
                # best = self.model
                print('new checkpoint with val loss: {}'.format(minimo))
                export(self.model, self.model_label, epoch)
        # plot_log(self.model_label, log)
        # self.model = best
        # export(self.model, self.model_label, epoch = None)

    def _train_loop(self, loader, optimizer, criterion):
        log = []
        # log = 0
        self.model.train()
        for batch in tqdm(loader):
            # raw, exam_id, label = batch
            raw = batch['X']
            h = batch['H'].to(self.device)
            ecg = get_inputs(raw, device = self.device)

            embedding = self.model.forward(ecg)
            loss = criterion(embedding, h)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log.append(loss.item())
            # log += loss.item()
        return log
        # return log / len(loader)
    
    def _eval_loop(self, loader, criterion):
        self.model = self.model.to(self.device)
        log = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                # raw, exam_id, label = batch
                raw = batch['X']
                h = batch['H'].to(self.device)
                ecg = get_inputs(raw, device = self.device)

                embedding = self.model.forward(ecg)
                loss = criterion(embedding, h)

                log += loss.item()
        return log / len(loader)