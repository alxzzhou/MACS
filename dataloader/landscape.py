import os
import random

import cv2
import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm


class Landscape(Dataset):
    def __init__(self, args):
        super(Landscape, self).__init__()
        image_lst = "image/"
        audio_lst = "audio/"

        self.image = os.path.join(args.data_dir, image_lst)
        self.audio = os.path.join(args.data_dir, audio_lst)
        self.image_path = list()
        self.audio_path = list()
        self.size = 512
        self.data_set = args.data_set
        self.input_length, self.sr = args.input_length, args.sr
        self.audio_only = not args.training
        self.cfg = args.cfg if hasattr(args, 'cfg') else False

        images = [file_path[:-4] for file_path in os.listdir(self.image)]
        audios = [file_path[:-4] for file_path in os.listdir(self.audio)]
        self.samples = set(images)

        self.prepare_dataset()
        self.audio_path = list(set(self.audio_path))
        self.num_samples = len(self.image_path)
        self._length = self.num_samples

    def __len__(self):
        return self._length

    def prepare_dataset(self):
        for id in tqdm(list(self.samples)):
            audio_id = id.split('_frame_')[0]
            self.image_path.append(os.path.join(self.image, id + ".jpg"))
            self.audio_path.append(os.path.join(self.audio, audio_id + ".wav"))

    def process_image(self, image):
        image = cv2.imread(image)
        img = np.array(image).astype(np.uint8)

        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((288, 512)),
            v2.RandomHorizontalFlip(),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        image = Image.fromarray(img)
        image = np.array(image).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sd_image = transforms(image)
        return sd_image

    def process_audio(self, aud, ):
        wav, sr = librosa.load(aud, sr=self.sr)
        if wav.shape[0] > sr * self.input_length:
            start = random.randint(0, wav.shape[0] - sr * self.input_length - 1)
            end = start + sr * self.input_length
            wav = wav[start:end]
        else:
            wav = np.pad(wav, (0, sr * self.input_length - wav.shape[0]), mode='constant', constant_values=(0, 0))
        wav = torch.from_numpy(wav)
        return wav

    def __getitem__(self, i):
        result = {}
        ytid = self.image_path[i % self.num_samples].split('/')[-1]
        ytid = ytid.split('_frame_')[0]
        result["ytid"] = ytid
        if not self.audio_only:
            image_path = self.image_path[i % self.num_samples]
            result["pixel_values"] = self.process_image(image_path)
        aud = os.path.join(self.audio, ytid + '.wav')
        result["audio_values"] = self.process_audio(aud)
        return result
