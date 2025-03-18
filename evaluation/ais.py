import argparse
import os
from datetime import datetime

import librosa.display
import numpy as np
import torch
import torch.nn.functional as F
import wav2clip
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

p = argparse.ArgumentParser()
p.add_argument('-d', '--dataset', type=str)
p.add_argument('-m', '--model_name', type=str, default='proposed')
p.add_argument('-i', '--image_path', type=str, )
p.add_argument('-a', '--audio_path', type=str, )
p.add_argument('--clip', type=str, default='openai/clip-vit-base-patch32')
args = p.parse_args()

audio_model = wav2clip.get_model()

vision_model = CLIPModel.from_pretrained(args.clip).cuda()
processor = CLIPProcessor.from_pretrained(args.clip)

dataset = args.dataset
model_name = args.model_name
image_path = args.image_path
audio_path = args.audio_path

ret, zret = [], []
imp = os.listdir(image_path)
imp = list(filter(lambda x: 'ipynb' not in x, imp))
aup = os.listdir(audio_path)

for fn in (pbar := tqdm(aup)):
    ytid = fn[:-4]
    img = os.path.join(image_path, f'{ytid}.jpg')
    if not os.path.exists(img): img = os.path.join(image_path, f'{ytid}.png')
    aud = os.path.join(audio_path, fn)

    track, _ = librosa.load(aud, sr=16000)
    image = Image.open(img)
    image = processor(images=image, return_tensors='pt').pixel_values.cuda()

    with torch.no_grad():
        image_features = vision_model.get_image_features(image)
        audio_features = wav2clip.embed_audio(track, audio_model)
        audio_features = torch.tensor(audio_features, device='cuda')

    sim = F.cosine_similarity(audio_features, image_features)
    r = sim.cpu().item()
    ret.append(r)

    calibration = []
    tmp = list(filter(lambda x: ytid not in x, imp))
    bsz = 100
    for i in range(len(tmp) // bsz + 1):
        l = []
        for s in tmp[i * bsz:i * bsz + bsz]:
            s = os.path.join(image_path, s)
            image = Image.open(s)
            image = processor(images=image, return_tensors='pt').pixel_values.cuda()
            l.append(image)

        images = torch.cat(l, dim=0)
        with torch.no_grad():
            image_features = vision_model.get_image_features(images)

        sim = F.cosine_similarity(audio_features, image_features)
        calibration.extend(list(sim.cpu().numpy()))
    calibration = np.array(calibration)
    mean, std = calibration.mean(), calibration.std()
    zret.append((r - mean) / std)

    pbar.set_description(f'score_mean = {sum(ret) / len(ret):.4f}, z_mean = {sum(zret) / len(zret): .4f}')

print(f'Final Result: mean_sim = {sum(ret) / len(ret):.4f}, z-score = {sum(zret) / len(zret):.4f}')

msg = f'\n{datetime.now()} - {sum(ret) / len(ret):.4f} - {sum(zret) / len(zret):.4f} - {dataset} - {model_name}'
with open('./ais.txt', 'a') as f:
    f.write(msg)
