import argparse
import math
import os.path
import warnings

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import AudioDataset
from modules import *

warnings.filterwarnings("ignore")


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    ps = argparse.ArgumentParser()
    ps.add_argument('--epochs', type=int, default=20)
    ps.add_argument('--batch_size', type=int, default=8)
    ps.add_argument('--lr', type=float, default=1e-3)
    ps.add_argument('--M', type=int, help='Number of splits in MixIT model.')

    ps.add_argument('--pt_path', type=str, help='Path to the pretrained weights. (Must be a directory)')
    ps.add_argument('--scheduler', type=str, default='cosine')
    ps.add_argument('--dataloader_num_workers', type=int, default=12)
    ps.add_argument('--gradient_accumulation_steps', type=int, default=8)
    ps.add_argument('--mixed_precision', type=str, default='fp16')
    ps.add_argument('--output_dir', type=str, )
    ps.add_argument('--finetune', action='store_true')
    ps.add_argument('--data_dir', type=str, )
    ps.add_argument('--dataset_name', type=str, )
    ps.add_argument('--input_length', type=int, default=8)
    ps.add_argument('--sr', type=int, default=16000)
    arg = ps.parse_args()

    arg.data_dir = os.path.join(arg.data_dir, arg.dataset_name)
    make_if_not_exist(arg.output_dir)

    assert arg.finetune or arg.dataset_name == 'FSD50K'
    assert not (arg.finetune and arg.dataset_name == 'FSD50K')
    return arg


args = parse_args()
logger = get_logger(__name__)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision
)

dataset = AudioDataset(args)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.dataloader_num_workers, pin_memory=True)
print(f'Size of Dataset = {len(dataset)}')

model = Separator(M=args.M, L=args.input_length, sr=args.sr).to(accelerator.device)
model.train()
model.load_clap()

if args.finetune:
    weight = torch.load(os.path.join(args.pt_path, 'separator_FSD50K_full.pt'))
    model.load_state_dict(weight)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

total_training_steps = args.epochs * math.ceil(len(dataloader) / args.gradient_accumulation_steps)
warmup_steps = total_training_steps // 10
scheduler = get_scheduler(
    args.scheduler, optimizer=optimizer,
    num_warmup_steps=warmup_steps * args.gradient_accumulation_steps,
    num_training_steps=total_training_steps * args.gradient_accumulation_steps
)

model, optimizer, dataloader, scheduler = accelerator.prepare(
    [model, optimizer, dataloader, scheduler]
)

total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

progress_bar = tqdm(range(total_training_steps), disable=not accelerator.is_local_main_process)

global_step = 0
for epoch in range(0, args.epochs):
    align_mean, rank_mean, mean = [], [], []
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            optimizer.zero_grad(set_to_none=True)
            audio = batch['audio_values'].to(accelerator.device)
            labels = batch['labels']
            loss = accelerator.unwrap_model(model)(audio, labels)
            snr = loss['mixit_loss']
            align = loss['align']
            rank = loss['rank']

            alpha = epoch / (args.epochs - 1)
            total_loss = (1 - alpha) * snr + alpha * (align + rank)
            accelerator.backward(total_loss)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            optimizer.step()
            scheduler.step()

        mean.append(snr.detach().item())
        align_mean.append(align.detach().item())
        rank_mean.append(rank.detach().item())
        logs = {
            'sdr': format(snr.detach().item(), '+.2f'),
            'rank': format(rank.detach().item(), '+.2f'),
            'rank_mean': format(sum(rank_mean) / len(rank_mean), '+.2f'),
            'align': format(align.detach().item(), '.4e'),
            'align_mean': format(sum(align_mean) / len(align_mean), '.4f'),
            'sdr_mean': format(sum(mean) / len(mean), '+.2f'),
            'lr': format(scheduler.get_last_lr()[0], '.2e')
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    path = os.path.join(args.output_dir, f'separator_{args.dataset_name}_full.pt')
    torch.save(accelerator.unwrap_model(model).state_dict(), path)
accelerator.end_training()
