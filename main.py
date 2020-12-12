import argparse
import torch
import wandb
import yaml
import os

from torch.utils.data import DataLoader
from scripts.evaluation_utils import load_model_by_name

import nomenclature

parser = argparse.ArgumentParser(description='Do stuff.')
parser.add_argument('--config_file', type = str, required = True)
parser.add_argument('--name', type = str, default = 'test')
parser.add_argument('--group', type = str, default = 'default')
parser.add_argument('--notes', type = str, default = '')
parser.add_argument("--mode", type = str, default = 'dryrun')
parser.add_argument('--epochs', type=int, default = 100)
parser.add_argument('--batch_size', type=int, default = 256)
parser.add_argument('--log_every', type = int, default = 5)
parser.add_argument('--eval_every', type = int, default = 5)
args = parser.parse_args()

cfg = yaml.load(open(f'{args.config_file}', 'rt'), Loader = yaml.FullLoader)
for key, value in cfg.items():
    args.__dict__[key] = value

os.environ['WANDB_MODE'] = args.mode
os.environ['WANDB_NAME'] = args.name
os.environ['WANDB_NOTES'] = args.notes

wandb.init(project = 'weak-gait', group = args.group)

wandb.config.update(vars(args))
wandb.config.update({'config': cfg})

if 'target_dataset' in args:
    dataset = nomenclature.DATASETS[args.target_dataset]
else:
    dataset = nomenclature.DATASETS[args.dataset]

train_dataloader = dataset.train_dataloader(args = args)
val_dataloader = dataset.val_dataloader(args = args)

model = nomenclature.MODELS[args.model](args)
if 'backbone' in args:
    state_dict = load_model_by_name(args.backbone)
    state_dict = {
        key.replace('module.', ''): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)

model.to(nomenclature.device)
print(model)
model = torch.nn.DataParallel(model)
wandb.watch(model)

trainer = nomenclature.TRAINERS[args.trainer](args, model)
trainer.train(train_dataloader, val_dataloader)
