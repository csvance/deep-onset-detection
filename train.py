import os
import plac
import random

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2 as l2r
import numpy as np
from model import residual_model
from callback import ValidationMetrics
from generator import SMBISequence


@plac.annotations(
    session=('Name of the training session. Used for saving weights / logs', 'option', 's', str),
    dims=('Base dimension of residual neural network', 'option', 'd', int),
    pooling=('Pooling operation to use (avg, max))', 'option', 'p', str),
    batch_size=('Batch size for training', 'option', 'b', int),
    epochs=('Max training epochs', 'option', 'e', int),
    normalization=('Training normalization strategy (batch, sample, global)', 'option', 'n', str),
    v_normalization=('Validation normalization strategy (batch, sample, global)', 'option', 'v', str),
    weights=('Weights path', 'option', 'w', str),
    workers=('Number of multiprocessing workers', 'option', 'W', int),
    l2=('L2 penalty', 'option', 'L', float),
    kernel_initializer=('Convolution weights initialization', 'option', 'I', str),
    random_seed=('RNG seed for NumPy + Tensorflow', 'option', 'S', int),
    cmd=('train, eval, salience', 'option', 'C', str)
)
def main(session: str,
         dims: int = 32,
         pooling: str = 'stride',
         batch_size: int = 16,
         epochs: int = 75,
         normalization: str = 'batch',
         v_normalization: str = 'global',
         weights=None,
         workers: int = 1,
         l2: float = 0.01,
         kernel_initializer: str = 'glorot_uniform',
         random_seed: int = None,
         cmd='train'):
    assert os.access('.', os.W_OK | os.R_OK)

    if cmd != 'salience':
        tf.compat.v1.disable_eager_execution()

    if random_seed is not None:
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        random.seed(random_seed)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if l2 is not None:
        kernel_regularizer = l2r(l2)
    else:
        kernel_regularizer = None

    model, loss_fns, metrics = residual_model(dims=dims,
                                              pooling=pooling,
                                              kernel_regularizer=kernel_regularizer,
                                              kernel_initializer=kernel_initializer)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    optimizer = SGD(lr=1e-4, momentum=0.9)

    model.compile(optimizer=optimizer, loss=loss_fns, metrics=metrics)

    try:
        os.mkdir('weights')
    except FileExistsError:
        assert os.access('weights', os.W_OK | os.R_OK)

    filepath = "weights/%s_epoch-{epoch:02d}_val_auc-{val_auc:.4f}.h5" % session
    checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')

    def sched(epoch):
        return 0.0001 * 0.5**(np.floor(epoch / 15))

    reduce_lr = LearningRateScheduler(schedule=sched)

    try:
        os.mkdir('logs')
    except FileExistsError:
        assert os.access('logs', os.W_OK | os.R_OK)

    if os.path.exists('logs/%s' % session):
        assert os.access('logs/%s' % session, os.W_OK | os.R_OK)

    tensorboard = TensorBoard(log_dir='logs/%s' % session, profile_batch=0)

    if cmd == 'train':
        Xy = np.load('data/train.npy', mmap_mode='r')

        X_train = Xy[:, :, 1:11]
        y_train = Xy[:, :, 0]
        d_train = np.load('data/train_props.npy')
        d_train = np.piecewise(d_train,
                               [
                                   d_train == 0,
                                   np.logical_and(0 < d_train, d_train <= 1 / 10),
                                   1 / 10 < d_train
                               ],
                               [
                                   0.95,
                                   lambda x: 10 * x,
                                   1
                               ])

        train_seq = SMBISequence(X=X_train,
                                 y=y_train,
                                 d=d_train,
                                 stage='train',
                                 batch_size=batch_size,
                                 normalization=normalization)

    X_test = np.swapaxes(np.load('data/test_inps.p', mmap_mode='r'), 1, 2)
    y_test = np.load('data/test_labels.p', mmap_mode='r')

    val_seq = SMBISequence(X=X_test,
                           y=y_test,
                           stage='test',
                           batch_size=batch_size,
                           normalization=v_normalization)

    validation_metrics = ValidationMetrics(seq=val_seq,
                                           batch_size=batch_size,
                                           seqlen=2000,
                                           n_feat=10)

    callbacks = [validation_metrics, checkpoint, reduce_lr, tensorboard]

    if cmd == 'train':
        model.fit_generator(train_seq,
                            validation_data=val_seq,
                            epochs=epochs,
                            callbacks=callbacks,
                            use_multiprocessing=True if workers != 1 else False,
                            workers=workers,
                            shuffle=True)

    if cmd == 'eval':
        if weights is not None:
            model.load_weights(weights)

        validation_metrics.model = model
        validation_metrics.on_epoch_end(0)

    if cmd == 'salience':
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        gidx = 0

        onset = False
        onset_idx = 0

        for bidx in range(0, len(val_seq)):
            mi, mo = val_seq.__getitem__(bidx)

            for sidx in range(0, batch_size):
                input_seq = mi[0][sidx]
                output_cls = mo[sidx][0]

                wrt = np.array([input_seq.astype(np.float32)])
                wrt_t = tf.constant(wrt)

                with tf.GradientTape() as g:

                    g.watch([wrt_t])
                    evaluated_cls = model([wrt_t])

                evaluated_gradients = np.abs(g.gradient(evaluated_cls, [wrt_t])[0][0].numpy())
                evaluated_gradients_all = np.sum(evaluated_gradients, axis=-1)

                if bool(output_cls):
                    if not onset:
                        onset = True
                        onset_idx = 1999
                else:
                    if onset:
                        onset = False

                fig = plt.figure(figsize=(10, 5), facecolor='#808080')
                plt.title('GTC seizure onset of slow activity prediction w/ saliency')
                plt.figtext(0.01, 0.01, 'Model: Residual CNN\nCarroll Vance <cs.vance@icloud.com>')

                ax = fig.gca()

                t = np.array([i for i in range(0, 2000)])

                xlim = (0, 2000)
                ylim = (0, np.max(evaluated_gradients_all))

                vmin = ylim[0]
                vmax = ylim[1]

                for i in range(0, 10):
                    plt.plot(t, input_seq[:, i])
                ax.pcolorfast(xlim, ax.get_ylim(), evaluated_gradients_all[np.newaxis], vmin=vmin, vmax=vmax,
                              cmap='viridis')
                plt.grid()

                locs = [i * 200 for i in range(0, 11)]
                labels = ["%d" % (int(loc / 200)) for loc in locs]

                plt.legend(['fp1-f7', 'f7-t7', 't7-p7', 'p7-o1', 'fp2-f8', 'f8-t8', 't8-p8', 'p8-o2', 'fz-cz', 'cz-pz'],
                           fontsize='small', loc='upper left')
                plt.xticks(locs, labels)
                plt.ylabel("Magnitude")
                plt.xlabel("Seconds")

                cb_pred = plt.colorbar(cm.ScalarMappable(cmap='viridis'), ticks=[0.0, 0.25, 0.5, 0.75, 1.0])

                if bool(np.round(evaluated_cls)) == bool(np.round(output_cls)):
                    textcolor = '#00EE00'
                    if bool(np.round(output_cls)):
                        outcome = 'TP'
                    else:
                        outcome = 'TN'
                else:
                    textcolor = 'red'
                    if bool(np.round(evaluated_cls)):
                        outcome = 'FP'
                    else:
                        outcome = 'FN'

                if onset:
                    cb_pred.ax.plot(0.5, evaluated_cls, '_', markersize=96, color='magenta', markeredgewidth=3)
                    cb_pred.ax.annotate("Positive", (0.5, 1.0), xytext=(-0.75, 1.02), color='red')
                else:
                    cb_pred.ax.plot(0.5, evaluated_cls, '_', markersize=96, color='magenta', markeredgewidth=3)
                    cb_pred.ax.annotate("Negative", (0.5, 0.0), xytext=(-0.75, -0.05), color='#00EE00')

                cb_pred.ax.set_ylabel('Prediction: %s' % outcome, color=textcolor)
                if onset:
                    plt.annotate("Onset", (onset_idx, 0), color='red', xytext=(onset_idx, np.max(input_seq)),
                                 arrowprops=dict(color='red', width=1., headwidth=6.))
                    onset_idx -= 20

                plt.show()
                # plt.savefig('figs/%d.png' % gidx, facecolor='#808080')

                plt.close('all')

                print(gidx)
                gidx += 1


if __name__ == '__main__':
    plac.call(main)
