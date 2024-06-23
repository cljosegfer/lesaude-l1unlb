
import torch
import torch.nn as nn
import argparse

from models.baseline import ResnetBaseline
from runners.train import Runner

# argparser
parser = argparse.ArgumentParser()
parser.add_argument('model_label', type = str, choices = ['code', 'code15', 'codenk', 'cpsc2018', 'ptbxl', 'ningbo'])
args = parser.parse_args()

print(args.model_label)
if args.model_label == 'code':
    from dataloaders.code import CODE as DS
    from dataloaders.code import CODEsplit as DSsplit

    epochs = 30
    n_classes = 6

if args.model_label == 'code15':
    from dataloaders.code15 import CODE as DS
    from dataloaders.code15 import CODEsplit as DSsplit

    epochs = 15
    n_classes = 6

if args.model_label == 'codenk':
    from dataloaders.codenk import CODE as DS
    from dataloaders.codenk import CODEsplit as DSsplit

    epochs = 15
    n_classes = 6

if args.model_label == 'cpsc2018':
    from dataloaders.cpsc2018 import CPSC2018 as DS
    from dataloaders.cpsc2018 import CPSC2018split as DSsplit

    epochs = 100
    n_classes = 8

if args.model_label == 'ptbxl':
    from dataloaders.ptbxl import PTBXL as DS
    from dataloaders.ptbxl import PTBXLsplit as DSsplit

    epochs = 80
    n_classes = 5

if args.model_label == 'ningbo':
    from dataloaders.ningbo import NINGBO as DS
    from dataloaders.ningbo import NINGBOsplit as DSsplit

    epochs = 25
    n_classes = 9

# init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

database = DS()
model = ResnetBaseline(n_classes = n_classes)
# model = torch.load('output/{}/partial.pt'.format(args.model_label, args.model_label))
runner = Runner(device = device, model = model, database = database, split = DSsplit, model_label = args.model_label)

# run
runner.train(epochs)
runner.eval()