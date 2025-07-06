import argparse
import os.path
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import StableDiffusion

warnings.filterwarnings('ignore')


def parse_args():
    ps = argparse.ArgumentParser()
    # model hyperparams
    ps.add_argument('--M', type=int)

    # functional params
    ps.add_argument('--prompt', type=str, default='')
    ps.add_argument('--negative_prompt', type=str, default='')
    ps.add_argument('--cfg', action='store_true')

    # model dirs
    ps.add_argument('--pt_path', type=str, help='Path to the pretrained weights. (Must be a directory)')
    ps.add_argument('--sd_path', type=str, help='Path to Stable Diffusion (local/huggingface)')

    # others
    ps.add_argument('--batch_size', type=int, default=1)
    ps.add_argument('--data_dir', type=str)
    ps.add_argument('--data_set', type=str, default='test')
    ps.add_argument('--output_dir', type=str, default='./output')
    ps.add_argument('--input_length', default=8)
    ps.add_argument('--sr', type=int, default=16000)
    ps.add_argument('--mixed_precision', default='fp16')
    ps.add_argument('--dataset_name', default='Landscape')
    ps.add_argument('--training', default=False)
    args = ps.parse_args()

    args.data_dir = os.path.join(args.data_dir, args.dataset_name, args.data_set)
    args.output_dir = os.path.join(args.output_dir, args.dataset_name)
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    return args


args = parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for fn in os.listdir(args.output_dir):
    file = os.path.join(args.output_dir, fn)
    os.remove(file)

DS = eval(args.dataset_name)
dataset = DS(args)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)

model = StableDiffusion(pt_path=args.pt_path, sd_path=args.sd_path, train=False, dataset=args.dataset_name,
                        M=args.M, cfg=args.cfg).to(device)

generated = []
sr, length = args.sr, args.input_length
for batch in tqdm(dataloader):
    audio = batch['audio_values'].to('cuda')
    ytid = batch['ytid'][0]
    if ytid in generated: continue
    generated.append(ytid)

    with torch.no_grad():
        image = model.infer(audio, prompt=args.prompt, negative_prompt=args.negative_prompt)
    image = (image.cpu().numpy() * 255).astype(np.uint8)
    image = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{args.output_dir}/{ytid}.jpg', image)
