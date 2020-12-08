from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch_optimizer import Lookahead
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import plac
import numpy as np

D = 16
BN_EPS = 1e-5
BN_MOM = 0.99
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-3
EPOCHS = 30
LR_MAX = 0.01

CFG = [
    {'repeat': 2, 'dim': int(1 * D), 'expand': 4, 'stride': 2, 'final': False},
    {'repeat': 3, 'dim': int(2 * D), 'expand': 4, 'stride': 2, 'final': False},
    {'repeat': 1, 'dim': int(4 * D), 'expand': 4, 'stride': 2, 'final': True},
]


class OnsetDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.accuracy = pl.metrics.Accuracy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):

        X = self.X[item]

        if len(self.y.shape) == 2:
            y_pos = len(np.where(self.y[item] == 1)[0])
            y_neg = len(self.y[item]) - y_pos
            y = 1 if y_pos > 0 else 0

            if y_pos == 0:
                w = 1.
            else:
                y_pos_pct = y_pos / (y_pos + y_neg)

                if y_pos_pct < 0.5:
                    w = min(2., 4 * y_pos / (y_pos + y_neg))
                else:
                    w = min(2., 4 * y_neg / (y_pos + y_neg))
        else:
            w = 1.
            y = self.y[item]

        X = X.transpose((1, 0)).astype(np.float32)
        w = np.array([w]).astype(np.float32)[0]
        y = int(y)
        return X, y, w


class ResnetBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, expand: int, stride: int = 1):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(c_in, eps=BN_EPS, momentum=1 - BN_MOM)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(c_in, int(c_in * expand), kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(int(c_in * expand), eps=BN_EPS, momentum=1 - BN_MOM)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(int(c_in * expand), int(c_in * expand), kernel_size=1, padding=0, bias=False,
                               stride=stride)

        self.bn3 = nn.BatchNorm1d(int(c_in * expand), eps=BN_EPS, momentum=1 - BN_MOM)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv1d(int(c_in * expand), c_out, kernel_size=3, padding=1, bias=False)

        self.stride = stride

    def forward(self, x):
        x_skip = x

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)

        # We always downsample on first layer in group so that we lose the minimal number of skip connections
        if self.stride != 1:
            return x

        return x + x_skip


class OnsetModule(LightningModule):
    def __init__(self, Xy_train, Xy_test):
        super().__init__()

        self.X_train, self.y_train = Xy_train
        self.X_test, self.y_test = Xy_test

        self._test_logits = []
        self._test_true = []

        self.blocks = nn.ModuleList()

        c_in = CFG[0]['dim']
        self.conv_stem = nn.Conv1d(in_channels=10, out_channels=c_in, kernel_size=3, stride=2, padding=1, bias=False)

        features = None
        for cfg in CFG:
            if not cfg['final']:
                block = ResnetBlock(c_in=c_in, c_out=cfg['dim'], expand=cfg['expand'], stride=cfg['stride'])
            else:
                features = cfg['dim'] * cfg['expand']
                block = ResnetBlock(c_in=c_in, c_out=features, expand=cfg['expand'], stride=cfg['stride'])

            self.blocks.append(block)

            for i in range(1, cfg['repeat'] - 1):
                block = ResnetBlock(c_in=cfg['dim'], c_out=cfg['dim'], expand=cfg['expand'])
                self.blocks.append(block)

            c_in = cfg['dim']

        self.bn_head = nn.BatchNorm1d(features, eps=BN_EPS, momentum=1 - BN_MOM)
        self.relu_head = nn.ReLU()
        self.pool_head = nn.AdaptiveMaxPool1d((1,))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(features, 2)

    def forward(self, x):

        # Batch-wise z-score
        if self.training:
            x = (x - torch.mean(x)) / torch.std(x)

        x = self.conv_stem(x)
        for block in self.blocks:
            x = block(x)

        x = self.bn_head(x)
        x = self.relu_head(x)
        x = self.pool_head(x)[:, :, 0]
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def training_step(self, batch, batch_nb):
        X, y_target, alpha = batch

        y = self.forward(X)
        loss = torch.mean(alpha*F.cross_entropy(y, y_target, reduction="none"))

        self.log('train_loss', loss, prog_bar=False, logger=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        self.log('momentum', self.optimizers().param_groups[0]['momentum'])

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y_target, w = batch
        y = self(X)

        loss = F.cross_entropy(y, y_target)

        self._test_logits.append(y[:, 1].detach().cpu().numpy())
        self._test_true.append(y_target.detach().cpu().numpy())

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log('val_loss', avg_loss, prog_bar=False, logger=True)

        self._test_logits = np.concatenate(self._test_logits, axis=0)
        self._test_true = np.concatenate(self._test_true, axis=0)

        self.log('val_auc', roc_auc_score(self._test_true, self._test_logits))

        self._test_true = []
        self._test_logits = []

        return {'val_loss': avg_loss}

    def test_step(self, batch, batch_nb):
        X, y_target, w = batch
        y = self(X)

        loss = F.cross_entropy(y, y_target)

        self._test_logits.append(y[:, 1].detach().cpu().numpy())
        self._test_true.append(y_target.detach().cpu().numpy())

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        self.log('test_loss', avg_loss, prog_bar=False, logger=True)

        self._test_logits = np.concatenate(self._test_logits, axis=0)
        self._test_true = np.concatenate(self._test_true, axis=0)

        self.log('test_auc', roc_auc_score(self._test_true, self._test_logits))

        self._test_true = []
        self._test_logits = []

        self._test_true = []
        self._test_logits = []

        return {'test_loss': avg_loss}

    def configure_optimizers(self):
        inner_optimizer = torch.optim.SGD(self.parameters(), lr=0.0001)
        optimizer = Lookahead(inner_optimizer)
        schedule = {'scheduler': OneCycleLR(inner_optimizer,
                                            max_lr=LR_MAX,
                                            epochs=EPOCHS,
                                            steps_per_epoch=int(len(self.X_train) // BATCH_SIZE),
                                            verbose=False),
                    'name': 'learning_rate',
                    'interval': 'step',
                    'frequency': 1
                    }
        return [optimizer], [schedule]

    def train_dataloader(self):
        return DataLoader(OnsetDataset(self.X_train, self.y_train),
                          shuffle=True,
                          drop_last=True,
                          batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(OnsetDataset(self.X_test, self.y_test),
                          shuffle=True,
                          batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return self.val_dataloader()


@plac.annotations()
def main():
    Xy = np.load('data/train.npy', mmap_mode='r')

    X_train = Xy[:, :, 1:11]
    y_train = Xy[:, :, 0]

    X_test = np.swapaxes(np.load('data/test_inps.p', mmap_mode='r'), 1, 2)
    y_test = np.load('data/test_labels.p', mmap_mode='r')

    model = OnsetModule((X_train, y_train), (X_test, y_test))

    trainer = Trainer(gpus=1,
                      precision=32,
                      max_epochs=EPOCHS,
                      log_every_n_steps=5,
                      flush_logs_every_n_steps=10)
    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    plac.call(main)
