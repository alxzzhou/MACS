import os
import random

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

filter_list = [
    'human sounds',
    'human voice',
    'whistling',
    'respiratory sounds',
    'human locomotion',
    'digestive',
    'hands',
    'heart sounds and heartbeat',
    'otoacoustic emission',
    'human group actions',
    'generic impact sounds',
    'surface contact',
    'deformable shell',
    'onomatopoeia',
    'silence',
    'other sourceless',
    'domestic animals and pets',
    'livestock and farm animals and working animals',
    'wild animals',
    'vehicle',
    'domestic sounds and home sounds',
    'mechanisms',
    'tools',
    'miscellaneous sources',
    'specific impact sounds',
    'sound reproduction',
    'accoustic environment',
    'musical instrument',
    'music genre',
    'musical concepts',
    'music role',
    'music mood',
    'sound reproduction'
]


class AudioDataset(Dataset):
    def __init__(self, args):
        super(AudioDataset, self).__init__()
        self.audio = args.data_dir
        self.input_length, self.sr, self.M = args.input_length, args.sr, args.M
        self.dataset_name = args.dataset_name
        if args.dataset_name != 'FSD50K':
            self.labels = self.get_labels()
        else:
            self.labels = self.get_labels_FSD50K()
        self.audio_path = self.prepare_dataset()

    def __len__(self):
        return len(self.audio_path)

    def get_labels(self):
        csv = pd.read_csv(f'./csvs/{self.dataset_name}.csv', sep='\t' if self.dataset_name != 'AudioSet' else ',',
                          encoding='utf-8')
        ret = {}
        for it in csv.iloc:
            l = it.event_labels.replace('_', ' ').split(',')[:self.M // 2]
            l = ['The sound of ' + i for i in l]
            if len(l) < self.M // 2: l.extend(['Noise'] * (self.M // 2 - len(l)))
            ret[it.filename if self.dataset_name == 'Landscape' else it.filename[:11]] = l
        return ret

    def get_labels_FSD50K(self):
        dev = pd.read_csv('./csvs/collection_dev.csv').dropna()
        eval = pd.read_csv('./csvs/collection_eval.csv').dropna()
        csv = pd.concat([dev, eval])

        def handle_label(l):
            prompts = [
                'The sound of ',
            ]
            l = l.replace('_', ' ').lower()
            sp = l.split(',')
            sp = list(filter(lambda x: x not in filter_list, sp))[:5]
            sp = [random.choice(prompts) + s for s in sp]
            if len(sp) < 5: sp.extend(['Noise'] * (5 - len(sp)))
            return sp

        ret = {}
        for it in csv.iloc:
            fn = str(int(it.fname))
            label = str(it.labels)
            ret[fn] = handle_label(label)
        return ret

    def prepare_dataset(self, ):
        ret = []
        keys = self.labels.keys()
        for fn in os.listdir(self.audio):
            if fn[:-4] not in keys: continue
            ret.append(os.path.join(self.audio, fn))
        random.shuffle(ret)
        return ret

    def process_audio(self, aud, ):
        wav, sr = librosa.load(aud, sr=self.sr)
        if wav.shape[0] > sr * self.input_length:
            start = random.randint(0, wav.shape[0] - sr * self.input_length - 1)
            end = start + sr * self.input_length
            wav = wav[start:end]
        else:
            padlen = sr * self.input_length - wav.shape[0]
            prefix = random.randint(0, padlen)
            postfix = padlen - prefix
            wav = np.pad(wav, (prefix, postfix), mode='constant', constant_values=(0, 0))
        wav = torch.from_numpy(wav)
        return wav

    def fetch(self, index):
        aud = self.audio_path[index]
        id = aud.split('/')[-1][:-4]
        aud = self.process_audio(aud)
        label = self.labels[id]
        return aud, label

    def __getitem__(self, i):
        result = {}
        audio, label = self.fetch(i)

        result['audio_values'] = torch.tensor(audio)
        result['labels'] = label
        return result
