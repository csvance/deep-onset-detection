import numpy as np
from tensorflow.keras.utils import Sequence
import random

from enum import Enum


class Normalization(Enum):
    NONE = 0
    BATCH = 1
    BATCH_F = 2
    SAMPLE = 3
    SAMPLE_F = 4
    GLOBAL = 5

    @staticmethod
    def from_str(s):
        if s == 'batch':
            return Normalization.BATCH
        elif s == 'batch_f':
            return Normalization.BATCH_F
        elif s == 'global':
            return Normalization.GLOBAL
        elif s == 'sample':
            return Normalization.SAMPLE
        elif s == 'sample_f':
            return Normalization.SAMPLE_F
        else:
            return Normalization.NONE


class SMBISequence(Sequence):

    GLOBAL_MEAN = 24.0057203800476806
    GLOBAL_STD = 521.5308728112200924

    def __init__(self,
                 X,
                 y,
                 d=None,
                 stage: str = "train",
                 batch_size: int = 128,
                 normalization=Normalization.BATCH,
                 augment: bool = False,
                 ):

        self.batch_size = batch_size
        self.stage = stage

        self.X = X
        self.y = y
        self.d = d

        self.normalization = Normalization.from_str(normalization)

        self.augment = augment

        self.sample_map = [i for i in range(0, len(self)*self.batch_size)]
        self.epochs = 0
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.stage == "train":
            random.shuffle(self.sample_map)

        self.epochs += 1

    def __getitem__(self, idx):

        batch_input = np.zeros((self.batch_size, self.X.shape[1], self.X.shape[2]))
        batch_discount = np.ones((self.batch_size, ), dtype=np.float32)
        batch_output_cls = np.zeros((self.batch_size, 1))

        for i in range(0, self.batch_size):
            sample_idx = self.sample_map[idx * self.batch_size + i]

            batch_input[i] = self.X[sample_idx, :, :]
            if self.d is not None:
                batch_discount[i] = self.d[sample_idx]
            batch_output_cls[i] = 1 if len(np.where(self.y[sample_idx] == 1)[0]) > 0 else 0

        if self.normalization == Normalization.BATCH:
            u = np.mean(batch_input)
            o = np.std(batch_input)

            batch_input -= u
            batch_input /= o
        elif self.normalization == Normalization.BATCH_F:
            batch_input -= np.mean(batch_input, axis=(0, 1))
            batch_input /= np.std(batch_input, axis=(0, 1))
        elif self.normalization == Normalization.SAMPLE:
            u = np.mean(batch_input, axis=(1, 2))
            o = np.std(batch_input, axis=(1, 2))
            for b in range(0, self.batch_size):
                batch_input[b] -= u[b]
                batch_input[b] /= o[b]
        elif self.normalization == Normalization.SAMPLE_F:
            u = np.mean(batch_input, axis=1)
            o = np.std(batch_input, axis=1)
            for b in range(0, self.batch_size):
                batch_input[b] -= u[b]
                batch_input[b] /= o[b]
        elif self.normalization == Normalization.GLOBAL:
            batch_input -= SMBISequence.GLOBAL_MEAN
            batch_input /= SMBISequence.GLOBAL_STD
        elif self.normalization == Normalization.NONE:
            pass
        else:
            raise ValueError

        return batch_input, batch_output_cls, batch_discount
