import matplotlib.pyplot as plt
import numpy as np
import plac
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sam import SAM

PLOT = True
HISTO = True

D = 32
SE = 4

BN_EPS = 0.001
BN_MOM = 0.01

EPOCHS = 2
BATCH_SIZE = 16
LR = 0.05

L2 = 0.00025
WEIGHT_DECAY = 0.
RHO = 0.1
DROPOUT = 0.5


class OnsetDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.astype(np.int64)

        if len(self.y.shape) == 2:
            self.w = compute_class_weight(y=np.max(self.y, axis=-1),
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

            y = 1 if y_pos else 0
            w = self.w[y]

            # Discount loss if either label is over 90% of the sequence
            if y:
                pct_pos = y_pos / (y_pos + y_neg)
                pct_neg = y_neg / (y_pos + y_neg)

                pct = min(pct_pos, pct_neg)
                if pct < 0.1:
                    w *= 10 * pct
                # Make up for discount
                w *= 1.1
        else:
            # When testing we only have a sequence wide label
            y = self.y[item]
            w = self.w[y]

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
    def __init__(self, c_in: int, c_out: int, stride: int = 1, project: bool = False, se: int = 1):
        super().__init__()

        self.norm1 = nn.BatchNorm1d(num_features=int(c_in), momentum=BN_MOM, eps=BN_EPS)
        self.relu1 = nn.ReLU()
        if stride == 2:
            self.blurpool = BlurPool1d(int(c_in))
        else:
            self.blurpool = None
        self.conv1 = nn.Conv1d(in_channels=c_in,
                               out_channels=c_in,
                               kernel_size=3,
                               padding=1,
                               bias=False)

        self.norm2 = nn.BatchNorm1d(num_features=c_in, momentum=BN_MOM, eps=BN_EPS)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=c_in,
                               out_channels=c_out,
                               kernel_size=3,
                               padding=1,
                               bias=False)

        if se > 1:
            self.se_pool = nn.AdaptiveAvgPool1d((1,))
            self.conv_squeeze = nn.Conv1d(c_in, int(c_in / se), kernel_size=1, bias=False)
            self.se_relu = nn.ReLU()
            self.conv_excite = nn.Conv1d(int(c_in / se), int(c_in), kernel_size=1, bias=False)
            self.se_sigmoid = nn.Sigmoid()
        else:
            self.se_pool = None
            self.conv_squeeze = None
            self.conv_excite = None
            self.se_relu = None
            self.se_sigmoid = None

        self.se = se
        self.stride = stride
        self.project = project

    def forward(self, x):
        x_skip = x

        x = self.norm1(x)
        x = self.relu1(x)
        if self.stride == 2:
            x = self.blurpool(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.relu2(x)
        if self.se > 1:
            x_se = self.se_pool(x)
            x_se = self.conv_squeeze(x_se)
            x_se = self.se_relu(x_se)
            x_se = self.conv_excite(x_se)
            x_se = self.se_sigmoid(x_se)
            x = x * x_se

        x = self.conv2(x)

        if self.project:
            return x

        if self.stride == 2:
            x_skip = self.blurpool(x_skip)

        return x + x_skip


class OnsetModule(pl.LightningModule):
    def __init__(self, Xy_train=None, Xy_valid=None, Xy_test=None,
                 epochs: int = EPOCHS,
                 lr: float = LR,
                 dropout: float = DROPOUT,
                 l2: float = L2,
                 rho: float = RHO,
                 weight_decay: float = WEIGHT_DECAY,
                 d: int = D,
                 se: int = SE,
                 seed: int = 0):

        super().__init__()
        self.save_hyperparameters('d', 'epochs', 'lr', 'l2', 'rho', 'weight_decay', 'dropout', 'se', 'seed')

        if Xy_train is not None:
            self.X_train, self.y_train = Xy_train
        else:
            self.X_train, self.y_train = None, None

        if Xy_valid is not None:
            self.X_valid, self.y_valid = Xy_valid
        else:
            self.X_valid, self.y_valid = None, None

        if Xy_test is not None:
            self.X_test, self.y_test = Xy_test
        else:
            self.X_test, self.y_test = None, None

        self._test_pred = []
        self._test_true = []

        self.blocks = nn.ModuleList()

        blocks = [
            {'repeat': 1, 'dim': int(1 * d), 'stride': 2, 'project': False, 'se': se},
            {'repeat': 1, 'dim': int(1 * d), 'stride': 2, 'project': False, 'se': se},
            {'repeat': 2, 'dim': int(2 * d), 'stride': 2, 'project': True, 'se': se},
            {'repeat': 1, 'dim': int(4 * d), 'stride': 2, 'project': True, 'se': se},
        ]

        c_in = blocks[0]['dim']
        self.conv_stem = nn.Conv1d(in_channels=10,
                                   out_channels=c_in,
                                   kernel_size=9,
                                   padding=4,
                                   bias=False)

        for cfg in blocks:
            block = ResnetBlock(c_in=c_in,
                                c_out=cfg['dim'],
                                stride=cfg['stride'],
                                project=cfg['project'],
                                se=cfg['se'])
            self.blocks.append(block)

            for i in range(1, cfg['repeat']):
                block = ResnetBlock(c_in=cfg['dim'],
                                    c_out=cfg['dim'],
                                    se=cfg['se'])
                self.blocks.append(block)

            c_in = cfg['dim']

        self.norm_head = nn.BatchNorm1d(num_features=c_in, momentum=BN_MOM, eps=BN_EPS)
        self.relu_head = nn.ReLU()
        self.pool_head = nn.AdaptiveMaxPool1d((1,))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, 2, bias=True)

        self._lr = lr
        self._epochs = epochs
        self._l2 = l2
        self._weight_decay = weight_decay
        self._rho = rho
        self._seed = seed

    def init(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, x: torch.tensor):
        if self.training:
            mu = torch.mean(x)
            sd = torch.std(x)
        else:
            mu = 19.2833522369622834
            sd = 530.5211141225101983
        x = (x - mu) / sd

        x = self.conv_stem(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)

        x = self.norm_head(x)
        x = self.relu_head(x)
        x = self.pool_head(x)[:, :, 0]
        x = self.dropout(x)
        y = self.fc(x)

        return y

    def training_step(self, batch, batch_nb):
        X, y_target, w = batch

        def forward_loss_backward():
            y = self.forward(X)
            step_loss = torch.mean(w * torch.unsqueeze(F.cross_entropy(y, y_target, reduction="none"), dim=-1))
            if self._l2 > 0.:
                l2 = None
                for p in self.parameters():
                    if p.requires_grad:
                        if l2 is None:
                            l2 = self._l2*torch.sum(torch.square(p))
                        else:
                            l2 += self._l2*torch.sum(torch.square(p))
                step_loss = step_loss + l2
            step_loss.backward()
            return step_loss

        loss = forward_loss_backward()
        opt: SAM = self.optimizers()
        opt.first_step(zero_grad=True)
        forward_loss_backward()
        opt.second_step(zero_grad=True)

        self.log('train_loss', loss, prog_bar=False, logger=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        self.log('momentum', self.optimizers().param_groups[0]['momentum'])

        return {'loss': loss}

    def optimizer_step(
        self,
        *args,
        epoch: int = None,
        batch_idx: int = None,
        optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
        **kwargs,
    ) -> None:
        optimizer_closure()

    def backward(self, loss, optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        pass

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

        try:
            self.log('val_auc', roc_auc_score(self._test_true, self._test_pred))

            if PLOT:
                cm = confusion_matrix(y_true=self._test_true.astype(np.int),
                                      y_pred=np.round(self._test_pred).astype(np.int),
                                      normalize='true')
                ax = sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', vmin=0, vmax=1.)
                ax.set_title('Confusion Matrix - Epoch %d' % self.current_epoch)
                plt.show()

        except ValueError:
            pass

        if HISTO:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if 'conv' in name or 'fc' in name:
                        self.logger.experiment.add_histogram(name, param, self.global_step)

        self._test_true = []
        self._test_pred = []

    def test_step(self, batch, batch_nb):
        X, y_target, w = batch
        y = self(X)

        loss = torch.mean(w * torch.unsqueeze(F.cross_entropy(y, y_target, reduction="none"), dim=-1))

        self._test_pred.append(F.softmax(y.detach(), dim=-1)[:, 1].cpu().numpy())
        self._test_true.append(y_target.detach().cpu().numpy())

        return {'test_loss': loss, }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        self.log('test_loss', avg_loss, prog_bar=False, logger=True)

        self._test_pred = np.concatenate(self._test_pred, axis=0)
        self._test_true = np.concatenate(self._test_true, axis=0)

        try:
            self.log('test_auc', roc_auc_score(self._test_true, self._test_pred))

            if PLOT:
                cm = confusion_matrix(y_true=self._test_true.astype(np.int),
                                      y_pred=np.round(self._test_pred).astype(np.int),
                                      normalize='true')
                ax = sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', vmin=0, vmax=1.)
                ax.set_title('Confusion Matrix - Test')
                plt.show()
        except ValueError:
            pass

        self._test_true = []
        self._test_pred = []

        self._test_true = []
        self._test_pred = []

    def configure_optimizers(self):

        optimizer = SAM(self.parameters(), base_optimizer=torch.optim.SGD, rho=self._rho,
                        lr=self._lr, weight_decay=self._weight_decay)
        inner_optimizer = optimizer.base_optimizer

        schedule = {'scheduler': OneCycleLR(inner_optimizer,
                                            max_lr=self._lr,
                                            epochs=self._epochs,
                                            steps_per_epoch=int(
                                                len(self.X_train) / BATCH_SIZE),
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
        if self.X_valid is not None:
            return DataLoader(OnsetDataset(self.X_valid, self.y_valid),
                              batch_size=BATCH_SIZE)
        return None

    def test_dataloader(self):
        if self.X_test is not None:
            return DataLoader(OnsetDataset(self.X_test, self.y_test),
                              batch_size=BATCH_SIZE)
        return None


@plac.annotations(seed=('Random seed', 'option', 'S', int),
                  k=('k-folds', 'option', 'k', int))
def main(seed: int = 0,
         k: int = 10):

    pyX = np.load('data/pyX.npy', mmap_mode='r')
    folds = np.array_split(np.unique(pyX[:, 0, 0]), k)

    for ki in range(0, k):
        pl.seed_everything(k)

        pids_train = []
        for n in range(ki, ki+k-1):
            n = n % k
            pids_train.extend(folds[n])
        pids_valid = []
        for n in range(ki+k-1, ki+k):
            n = n % k
            pids_valid.extend(folds[n])

        idx_train = np.isin(pyX[:, 0, 0], pids_train)
        idx_valid = np.isin(pyX[:, 0, 0], pids_valid)

        X_train = pyX[idx_train, :, 2:]
        y_train = pyX[idx_train, :, 1]

        X_valid = pyX[idx_valid, :, 2:]
        y_valid = pyX[idx_valid, :, 1]

        d = D
        dropout = DROPOUT
        lr = LR
        se = SE
        epochs = EPOCHS
        rho = RHO

        model = OnsetModule((X_train, y_train),
                            (X_valid, y_valid),
                            d=d,
                            epochs=epochs,
                            lr=lr,
                            dropout=dropout,
                            se=se,
                            rho=rho,
                            seed=seed)

        model.init()
        logger = TensorBoardLogger('lightning_logs', name='seed_%d' % seed, default_hp_metric=True)
        cb_checkpoint = pl.callbacks.ModelCheckpoint(dirpath='checkpoint',
                                                     filename='seed_%d' % seed,
                                                     monitor='val_auc',
                                                     mode='max',
                                                     verbose=True)

        trainer = pl.Trainer(gpus=1,
                             precision=32,
                             max_epochs=epochs,
                             log_every_n_steps=5,
                             flush_logs_every_n_steps=5,
                             callbacks=[cb_checkpoint],
                             deterministic=True,
                             val_check_interval=0.1,
                             logger=logger)

        trainer.fit(model)
        logger.log_hyperparams(model.hparams, {'hp_metric': cb_checkpoint.best_model_score})

    """
    X_test = np.swapaxes(np.load('data/test_inps.p', mmap_mode='r'), 1, 2)
    y_test = np.load('data/test_labels.p', mmap_mode='r')

    results = trainer.test(ckpt_path=cb_checkpoint.best_model_path)
    logger.log_hyperparams(model.hparams, {'hp_metric': results[0]['test_auc']})
    logger.close()
    """


if __name__ == '__main__':
    plac.call(main)
