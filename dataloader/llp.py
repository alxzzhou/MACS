import os
import random

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm


class LLP(Dataset):
    def __init__(self, args):
        super(LLP, self).__init__()
        image_lst = "image/"
        audio_lst = "audio/"

        self.image = os.path.join(args.data_dir, image_lst)
        self.audio = os.path.join(args.data_dir, audio_lst)
        self.image_path = list()
        self.audio_path = list()
        self.bsz = args.batch_size
        self.size = 512
        self.data_set = args.data_set
        self.input_length, self.sr = args.input_length, args.sr
        self.audio_only = not args.training
        self.cfg = args.cfg if hasattr(args, 'cfg') else False

        images = [file_path[:-4] for file_path in os.listdir(self.image)]
        audios = [file_path[:-4] for file_path in os.listdir(self.audio)]
        self.samples = set(images)

        self.prepare_dataset()
        self.num_samples = len(self.image_path)
        self._length = self.num_samples

        # aspect ratio
        if not self.audio_only:
            self.resolution_step = 32
            self.resolutions, self.ar = self.get_resolutions_and_ar()
            self.ar_dict = self.get_ar_dict()
            self.idx_mapping, self.res_mapping = self.get_mapping_table()

            print(f"{args.data_set}, num samples: {self.num_samples}")

    def __len__(self):
        return self._length

    def get_resolutions_and_ar(self):
        base_res = self.size
        res = [base_res + self.resolution_step * i for i in range(-9, -1, 2)]
        ret = [(base_res, base_res)]
        for r in res:
            ret.append((r, base_res))
            ret.append((base_res, r))
        return ret, [r[0] / r[1] for r in ret]

    def get_ar_dict(self):
        ret = {r: [0, []] for r in self.resolutions}
        for im in tqdm(self.image_path, desc='Extracting AR Dict'):
            h, w = cv2.imread(im).shape[:2]
            idx = min(list(range(len(self.ar))), key=lambda x: abs(self.ar[x] - h / w))
            ret[self.resolutions[idx]][0] += 1
            ret[self.resolutions[idx]][1].append(im)
        for k in ret.keys():
            random.shuffle(ret[k][1])
        return ret

    def get_mapping_table(self):
        ret = []
        residuals = []
        transforms = []
        for r in self.ar_dict.keys():
            n = self.ar_dict[r][0]
            residual = n % self.bsz
            ret.extend(self.ar_dict[r][1][:-residual])
            residuals.extend(self.ar_dict[r][1][-residual:])
            transforms.extend([r] * (n - residual))
        ret.extend(residuals)
        transforms.extend([(self.size, self.size)] * len(residuals))
        return ret, transforms

    def prepare_dataset(self):
        for id in tqdm(list(self.samples)):
            audio_id = id.split('_frame_')[0]
            self.image_path.append(os.path.join(self.image, id + ".jpg"))
            self.audio_path.append(os.path.join(self.audio, audio_id + ".wav"))

    def process_image(self, image, resolution=(288, 512)):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            v2.Resize(resolution),
            v2.RandomHorizontalFlip(),
        ])

        sd_image = transforms(image)
        return sd_image

    def process_audio(self, aud, ):
        wav, sr = librosa.load(aud, sr=self.sr)
        if wav.shape[0] > sr * self.input_length:
            start = random.randint(0, wav.shape[0] - sr * self.input_length - 1)
            end = start + sr * self.input_length
            wav = wav[start:end]
        else:
            wav = np.pad(wav, (0, sr * self.input_length - wav.shape[0]), mode='reflect')
        wav = torch.from_numpy(wav)
        return wav

    def __getitem__(self, i):
        result = {}
        audio_path = self.audio_path[i]
        ytid = audio_path.split('/')[-1][:-4]

        if not self.audio_only:
            image_path, resolution = self.idx_mapping[i], self.res_mapping[i]
            result["pixel_values"] = self.process_image(image_path, resolution)
            audio_path = image_path.replace('image', 'audio')[:-4] + '.wav'
            ytid = audio_path.split('/')[-1][:-4]
        result["audio_values"] = self.process_audio(audio_path)
        result['ytid'] = ytid
        return result
