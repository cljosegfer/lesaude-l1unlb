
import torch
import torch.nn as nn
import argparse

from models.baseline import ResnetBaseline
from runners.pretrain import Runner
from dataloaders.code import CODE as DS
from dataloaders.code import CODEsplit as DSsplit

# init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 50
model_label = 'backbone'

database = DS()
model = ResnetBaseline(n_classes = 768)
runner = Runner(device = device, model = model, database = database, split = DSsplit, model_label = model_label)
# model = torch.load('output/{}/{}.pt'.format(args.model_label, args.model_label))

# run
runner.train(epochs)