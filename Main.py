import numpy as np

import torch
import torchvision

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Loader import load_ihdp, load_mimic
from Helper import TTrainer, FTrainer, PreTrainer
# from Models import FTransformer
from Models import PModel, VModel
from Models import FTransformer
# from MimicModel import FTransformer

def main():
    D_train, D_test = load_ihdp()
    # D_train, D_test = load_mimic()

    pmodel = PModel()
    # optimizer_p = Adam(pmodel.parameters(), lr=0.001, weight_decay=0.0001)
    # scheduler_p = CosineAnnealingWarmRestarts(optimizer_p, T_0=10, T_mult=1)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # ttrainer = TTrainer(model=pmodel, optimizer=optimizer_p, scheduler=scheduler_p, D_train=D_train, 
    #                     D_test=D_test, loss_fn=loss_fn, outname='pmodel', n_exp=100)
    # bestacc = ttrainer.train(root='/Users/Workspace/', device='mps')
    # checkpoint = torch.load('C:\\Workspace\\pmodel_mimiciii')
    checkpoint = torch.load('/Users/Workspace/Pmodel')
    pmodel.load_state_dict(checkpoint['model_state_dict'])

    ptmodel = FTransformer(num_layers=4, hidden=768)
    checkpoint = torch.load('/Users/Workspace/ptmodel')
    # checkpoint = torch.load('C:\\Workspace\\ptmodel_mimiciii')
    ptmodel.load_state_dict(checkpoint['model_state_dict'])
    # optimizer_pre = Adam(ptmodel.parameters(), lr=0.005, weight_decay=0.0001)
    # # # optimizer_pre = SGD(ptmodel.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)
    # scheduler_pre = CosineAnnealingWarmRestarts(optimizer_pre, T_0=3000, T_mult=1)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # trainer = PreTrainer(ptmodel, optimizer_pre, scheduler_pre, D_train, D_test, 
                        #  loss_fn, pmodel=pmodel, n_exp=100, outname='ptmodel')
    # bestmse = trainer.train(root='/Users/Workspace/', n_epoches=50, device='mps')

    # trainmodel = VModel()
    optimizer_f = Adam(ptmodel.parameters(), lr=0.005, weight_decay=0.0001)
    # # optimizer = SGD(ptmodel.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
    scheduler_f = CosineAnnealingWarmRestarts(optimizer_f, T_0=3000, T_mult=1)
    loss_fn = torch.nn.MSELoss()
    trainer = FTrainer(ptmodel, optimizer_f, scheduler_f, D_train, D_test, loss_fn, n_exp=100)
    bestmse = trainer.train(root='/Users/Workspace/', n_epoches=100)

if __name__ == '__main__':
    main()