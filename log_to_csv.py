from tensorboard.backend.event_processing import event_accumulator
import plac
import os
import numpy as np
import pandas as pd

TAGS = ['epoch_val_loss', 'epoch_val_acc', 'epoch_val_bce', 'epoch_val_sensitivity', 'epoch_val_auc', 'epoch_val_specificity']


@plac.annotations(
    session=('Session name', 'positional', None, str)
)
def main(session: str):

    runs_best = {}
    runs_all = {}
    for tag in TAGS:
        runs_best[tag] = []

    for i in range(0, 10):

        log_dir = os.path.join('logs', '%s_iter_%d' % (session, i))
        print('Scanning: %s' % log_dir)

        for root, dirs, files in os.walk(log_dir, topdown=False):
            for name in files:
                log_path = os.path.join(root, name)
                print('Reading Log: %s' % log_path)

                ea = event_accumulator.EventAccumulator(
                    log_path,
                    size_guidance={  # see below regarding this argument
                        event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                        event_accumulator.IMAGES: 4,
                        event_accumulator.AUDIO: 4,
                        event_accumulator.SCALARS: 0,
                        event_accumulator.HISTOGRAMS: 1})

                ea.Reload()

                auc = []
                for x in ea.Scalars('epoch_val_auc'):
                    auc.append(x.value)

                best_idx = np.argmax(auc)

                for tag in TAGS:
                    runs_best[tag].append(ea.Scalars(tag)[best_idx].value)


    df = pd.DataFrame.from_dict(runs_best)
    print("--mean--")
    print(df.mean())
    print("--std--")
    print(df.std())
    print("--argmax--")
    print(df.idxmax())
    print(df.iloc[np.argmax(df['epoch_val_auc'])])


if __name__ == '__main__':
    plac.call(main)
