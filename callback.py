from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import numpy as np


class ValidationMetrics(Callback):
    def __init__(self, seq, batch_size: int, seqlen: int, n_feat: int):
        Callback.__init__(self)

        self.seq = seq

        self.batch_size = batch_size
        self.seqlen = seqlen
        self.n_feat = n_feat

        self.X = np.zeros((self.batch_size*len(self.seq), self.seqlen, self.n_feat))
        self.y = np.zeros((self.batch_size*len(self.seq), 1))
        self.d = np.ones((self.batch_size*len(self.seq), 1), dtype=np.float32)

        idx = 0
        for bidx in range(0, len(seq)):

            bin, bout, _ = seq.__getitem__(bidx)

            self.X[idx:idx+batch_size] = bin
            self.y[idx:idx+batch_size] = bout

            idx += batch_size

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_predict_end(self, logs=None):
        self.on_epoch_end()

    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict([self.X, self.d])

        val_auc = roc_auc_score(self.y, y_pred)

        cm = confusion_matrix(self.y, np.round(y_pred))
        tn, fp, fn, tp = cm.ravel()

        val_sen = tp / (tp + fn)
        val_spe = tn / (tn + fp)

        print("\n\nval_auc: %.5f" % val_auc)
        print("val_sensitivity: %.5f" % val_sen)
        print("val_specificity: %.5f" % val_spe)
        print("")
        print("--Confusion Matrix--")
        print(cm)

        logs['val_auc'] = val_auc
        logs['val_sensitivity'] = val_sen
        logs['val_specificity'] = val_spe

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
