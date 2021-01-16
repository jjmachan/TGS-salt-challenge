import os
import time
import math
from random import randint
import cv2
import numpy as np
import torch


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def load_mask(_id, mask_dir):
    # mask_dir = '../data/competition_data/train/masks/'
    mask = cv2.imread(os.path.join(mask_dir, _id+'.png'))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)//255
    return mask.sum()


def stratified_split(df, sample_percent=0.25, seed=42):
    "Make a split on a df where cuts are provided"
    grouped = df.groupby('cuts')
    samples = []
    for name, group in grouped:
        sample_size = int(group.shape[0]*sample_percent)
        samples.extend(list(group.sample(sample_size, random_state=seed).index))
    return samples


def sample(dataset, num_samples):
    "sample a dataset object num_samples times"
    samples = []
    for i in range(num_samples):
        n = randint(0, len(dataset))
        img, mask = dataset[n]
        samples.append((img.squeeze(), mask.squeeze()))
    return samples


class EarlyStopping:
    def __init__(
            self, patience=7,
            verbose=False, delta=0,
            mode='min', path='checkpoint.pt',
            trace_func=print
            ):
        """

        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_prev = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        assert mode in ['max', 'min']
        if mode == 'max':
            self.multiple = 1
        else:
            self.multiple = -1

    def __call__(self, score, model):

        score = self.multiple*score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose and self.multiple == -1:
            self.trace_func(f'Score decreased ({self.score_prev*self.multiple:.6f} --> {score*self.multiple:.6f}).  Saving model ...')
        elif self.verbose and self.multiple == 1:
            self.trace_func(f'Score increased ({self.score_prev:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.score_prev = score
