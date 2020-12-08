from torch.utils.data import Dataset, DataLoader
from torch_optimizer import Lookahead
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import plac
import numpy as np

D = 32
BN_EPS = 1e-5
BN_MOM = 0.99
BATCH_SIZE = 16
WEIGHT_DECAY = 0.
L2 = 0.01
EPOCHS = 75
LR_MAX = 0.001

CFG = [
    {'repeat': 2, 'dim': int(1 * D), 'expand': 2, 'stride': 2, 'final': False},
    {'repeat': 3, 'dim': int(2 * D), 'expand': 2, 'stride': 2, 'final': False},
    {'repeat': 1, 'dim': int(4 * D), 'expand': 1, 'stride': 2, 'final': True},
]


class OnsetDataset(Dataset):
    def __init__(self, X, y, training: bool = True):
        self.X = X
        self.y = y.astype(np.int64)

        self.training = training

        if len(self.y.shape) == 2:
            self.w = compute_class_weight(y=[1 if len(np.where(yi)[0]) > 0 else 0 for yi in self.y],
                                          class_weight="balanced",
                                          classes=[0, 1])
        else:
            self.w = compute_class_weight(y=self.y,
                                          class_weight="balanced",
                                          classes=[0, 1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):

        X = self.X[item]

        if len(self.y.shape) == 2:
            y_pos = len(np.where(self.y[item] == 1)[0])
            y_neg = len(self.y[item]) - y_pos
            y = 1 if y_pos > 0 else 0

            if y_pos == 0:
                w = self.w[0]
            else:
                y_pos_pct = y_pos / (y_pos + y_neg)
                y_neg_pct = y_neg / (y_pos + y_neg)

                pct = min(y_pos_pct, y_neg_pct)

                if pct < 0.1 or pct > 0.9:
                    w = 0.1*self.w[1]
                else:
                    w = 1.2*self.w[1]
        else:
            y = self.y[item]
            w = self.w[[0, 1].index(y)]

        # Don't discount loss if not training
        if not self.training:
            w = self.w[[0, 1].index(y)]

        X = X.transpose((1, 0)).astype(np.float32)
        w = np.array([w]).astype(np.float32)
        return X, y, w


class BlurPool1d(nn.Module):
    def __init__(self, channels, blur_kernel_size: int = 3):
        super().__init__()

        self.channels = channels
        self.blur_kernel_size = blur_kernel_size

        if self.blur_kernel_size == 3:
            binomial = [1, 2, 1]
        elif self.blur_kernel_size == 5:
            binomial = [1, 4, 6, 4, 1]
        elif self.blur_kernel_size == 7:
            binomial = [1, 6, 15, 20, 15, 6, 1]
        else:
            raise ValueError('Supported kernel sizes are in {3, 5, 7}')

        bk = binomial

        bk = bk / np.sum(bk)
        bk = np.repeat(bk, self.channels)
        bk = np.reshape(bk, (self.blur_kernel_size, self.channels, 1))

        # WxCx1 -> Cx1xW
        bk = bk.transpose((1, 2, 0))

        self.kernel = nn.Parameter(torch.from_numpy(bk.astype(np.float32)), requires_grad=False)

    def forward(self, x):
        same = int(self.blur_kernel_size / 2)
        x = F.conv1d(x,
                     weight=self.kernel,
                     padding=same,
                     stride=2,
                     groups=self.channels)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, expand: int, stride: int = 1):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(c_in, eps=BN_EPS, momentum=1 - BN_MOM)
        self.relu1 = nn.ReLU()
        if stride == 2:
            self.blurpool = BlurPool1d(int(c_in))
        else:
            self.blurpool = None
        self.conv1 = nn.Conv1d(c_in, int(c_in * expand), kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(int(c_in * expand), eps=BN_EPS, momentum=1 - BN_MOM)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(int(c_in * expand), int(c_in * expand), kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(int(c_in * expand), eps=BN_EPS, momentum=1 - BN_MOM)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv1d(int(c_in * expand), c_out, kernel_size=3, padding=1, bias=False)

        self.stride = stride

    def forward(self, x):
        x_skip = x

        x = self.bn1(x)
        x = self.relu1(x)
        if self.stride == 2:
            x = self.blurpool(x)
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

        self._test_pred = []
        self._test_true = []

        self.blocks = nn.ModuleList()

        c_in = CFG[0]['dim']
        self.conv_stem = nn.Conv1d(in_channels=10, out_channels=c_in, kernel_size=9, padding=4, bias=False)

        features = None
        for cfg in CFG:
            if not cfg['final']:
                block = ResnetBlock(c_in=c_in, c_out=cfg['dim'], expand=cfg['expand'], stride=cfg['stride'])
            else:
                features = cfg['dim']
                block = ResnetBlock(c_in=c_in, c_out=features, expand=cfg['expand'], stride=cfg['stride'])

            self.blocks.append(block)

            for i in range(1, cfg['repeat'] - 1):
                block = ResnetBlock(c_in=cfg['dim'], c_out=cfg['dim'], expand=cfg['expand'])
                self.blocks.append(block)

            c_in = cfg['dim']

        self.bn_head = nn.BatchNorm1d(features, eps=BN_EPS, momentum=1 - BN_MOM)
        self.relu_head = nn.ReLU()
        self.pool_head = nn.AdaptiveMaxPool1d((1,))
        self.fc = nn.Linear(int(2*features), 2, bias=False)

    def init(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, x):

        # Batch-wise z-score
        if self.training:
            mu = torch.mean(x)
            sd = torch.std(x)
        else:
            mu = torch.mean(x, dim=(1, 2)).unsqueeze(dim=-1).unsqueeze(dim=-1)
            sd = torch.std(x, dim=(1, 2)).unsqueeze(dim=-1).unsqueeze(dim=-1)

        x = (x - mu) / sd

        x = self.conv_stem(x)
        for block in self.blocks:
            x = block(x)

        x = self.bn_head(x)
        x = self.relu_head(x)
        x = F.max_pool1d(x, kernel_size=125, stride=125, padding=0)
        x = x.view((x.size(0), x.size(1)*x.size(2)))
        y = self.fc(x)

        return y

    def training_step(self, batch, batch_nb):
        X, y_target, w = batch

        y = self.forward(X)
        loss = torch.mean(w * torch.unsqueeze(F.cross_entropy(y, y_target, reduction="none"), dim=-1))

        if L2 > 0.:
            l2 = torch.sum(L2 * torch.pow(self.conv_stem.weight, 2))
            for block in self.blocks:
                l2 += torch.sum(L2 * torch.pow(block.conv1.weight, 2))
                l2 += torch.sum(L2 * torch.pow(block.conv2.weight, 2))
                l2 += torch.sum(L2 * torch.pow(block.conv3.weight, 2))

            loss += l2

        self.log('train_loss', loss, prog_bar=False, logger=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        self.log('momentum', self.optimizers().param_groups[0]['momentum'])

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y_target, w = batch
        y = self(X)

        loss = torch.mean(w * torch.unsqueeze(F.cross_entropy(y, y_target, reduction="none"), dim=-1))

        self._test_pred.append(F.softmax(y.detach(), dim=-1)[:, 1].cpu().numpy())
        self._test_true.append(y_target.detach().cpu().numpy())

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log('val_loss', avg_loss, prog_bar=False, logger=True)

        self._test_pred = np.concatenate(self._test_pred, axis=0)
        self._test_true = np.concatenate(self._test_true, axis=0)

        self.log('val_auc', roc_auc_score(self._test_true, self._test_pred))

        tn, fp, fn, tp = confusion_matrix(y_true=self._test_true.astype(np.int),
                                          y_pred=np.round(self._test_pred).astype(np.int)).ravel()

        print("\n\n----Confusion Matrix----")
        print("\t[neg\t\tpos\t\t]")
        print("neg\t[%d\t\t%d\t]" % (tn, fp))
        print("pos\t[%d\t\t%d\t]\n" % (fn, tp))

        self._test_true = []
        self._test_pred = []

    def test_step(self, batch, batch_nb):
        X, y_target, w = batch
        y = self(X)

        loss = torch.mean(w * torch.unsqueeze(F.cross_entropy(y, y_target, reduction="none"), dim=-1))

        self._test_pred.append(F.softmax(y.detach(), dim=-1)[:, 1].cpu().numpy())
        self._test_true.append(y_target.detach().cpu().numpy())

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        self.log('test_loss', avg_loss, prog_bar=False, logger=True)

        self._test_pred = np.concatenate(self._test_pred, axis=0)
        self._test_true = np.concatenate(self._test_true, axis=0)

        self.log('test_auc', roc_auc_score(self._test_true, self._test_pred))

        tn, fp, fn, tp = confusion_matrix(y_true=self._test_true.astype(np.int),
                                          y_pred=np.round(self._test_pred).astype(np.int)).ravel()

        print("\n\n----Confusion Matrix----")
        print("\t[neg\t\tpos\t\t]")
        print("neg\t[%d\t\t%d\t]" % (tn, fp))
        print("pos\t[%d\t\t%d\t]\n" % (fn, tp))

        self._test_true = []
        self._test_pred = []

        self._test_true = []
        self._test_pred = []

    def configure_optimizers(self):
        inner_optimizer = torch.optim.SGD(self.parameters(),
                                          lr=0.001,
                                          momentum=0.9)
        optimizer = Lookahead(inner_optimizer)
        schedule = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=15, gamma=0.5)
        return [optimizer], [schedule]

    def train_dataloader(self):
        return DataLoader(OnsetDataset(self.X_train, self.y_train),
                          shuffle=True,
                          drop_last=True,
                          batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(OnsetDataset(self.X_test, self.y_test, training=False),
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
    model.init()

    trainer = Trainer(gpus=1,
                      precision=32,
                      max_epochs=EPOCHS,
                      log_every_n_steps=5,
                      flush_logs_every_n_steps=10)
    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    plac.call(main)
