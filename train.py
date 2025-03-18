import argparse
import math
import os.path
import warnings

import torch
from accelerate import Accelerator
from diffusers import get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import *

warnings.filterwarnings("ignore")


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    ps = argparse.ArgumentParser()

    # regular
    ps.add_argument('--epochs', type=int, default=15)
    ps.add_argument('--batch_size', type=int, default=2)
    ps.add_argument('--lr', type=float, default=1e-4)
    ps.add_argument('--scheduler', type=str, default='cosine')
    ps.add_argument('--gradient_accumulation_steps', type=int, default=8)
    ps.add_argument('--mixed_precision', type=str, default='fp16')
    ps.add_argument('--output_dir', type=str, )

    # model hyperparams
    ps.add_argument('--M', type=int, help='Number of splits in separator.')

    # functional params
    ps.add_argument('--cfg', action='store_true')

    # model dirs
    ps.add_argument('--pt_path', type=str, help='Path to the pretrained weights. (Must be a directory)')
    ps.add_argument('--sd_path', type=str, help='Path to Stable Diffusion (local/huggingface)')

    # dataloader params
    ps.add_argument('--dataloader_num_workers', type=int, default=12)
    ps.add_argument('--data_dir', type=str, )
    ps.add_argument('--dataset_name', type=str, )
    ps.add_argument('--data_set', type=str, default='train')
    ps.add_argument('--input_length', type=int, default=8)
    ps.add_argument('--sr', type=int, default=16000)
    ps.add_argument('--training', default=True)
    ps.add_argument('--output_size', default=512)
    arg = ps.parse_args()

    arg.csv_path = arg.csv_path.format(arg.dataset_name) if arg.csv_path is not None else None
    arg.data_dir = os.path.join(arg.data_dir, arg.dataset_name, arg.data_set)
    make_if_not_exist(arg.output_dir)
    return arg


args = parse_args()

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision
)

DS = eval(args.dataset_name)
dataset = DS(args)

dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True if args.dataset_name == 'Landscape' else False,
                        num_workers=args.dataloader_num_workers, pin_memory=True)

dtype = torch.float32
if accelerator.mixed_precision == 'fp16':
    dtype = torch.float16

model = StableDiffusion(pt_path=args.pt_path, sd_path=args.sd_path, dataset=args.dataset_name, M=args.M).to(
    accelerator.device)

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

global_step, accuracy = 0, []
for epoch in range(0, args.epochs):
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            optimizer.zero_grad(set_to_none=True)
            image = batch['pixel_values'].to(accelerator.device)
            audio = batch['audio_values'].to(accelerator.device)
            output = accelerator.unwrap_model(model)(audio, image)
            loss = output['sd_loss']

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(model.parameters(), 1.)
                progress_bar.update(1)
                global_step += 1

            optimizer.step()
            scheduler.step()

        losses = {k: v.detach().item() for k, v in output.items() if 'loss' in k}
        logs = {'lr': scheduler.get_last_lr()[0]}
        logs.update(losses)
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    path = os.path.join(args.output_dir, f'{args.dataset_name}_mlp.pt')
    torch.save(accelerator.unwrap_model(model).mlp.state_dict(), path)
    path = os.path.join(args.output_dir, f'{args.dataset_name}_unet.pt')
    torch.save(accelerator.unwrap_model(model).unet.state_dict(), path)
accelerator.end_training()
