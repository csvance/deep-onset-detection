import plac
import pandas as pd
import numpy as np
import random

# ECG Sampling Frequency
HZ = 200

# ECG Length (seconds)
LENGTH = 180

# Number of samples per episode
EP_SAMPLES = 36001

# Length of onset
ONSET = HZ * 10

# Number of samples to create per episode
SAMPLES_PER_EPISODE = 1000


@plac.annotations(
    csv_path=('Path to the csv file', 'option', 'i', str),
    seqlen=('Length of the sequence to generate', 'option', 'l', int)
)
def main(csv_path: str = 'data/train_full_seq.csv',
         seqlen: int = int(HZ*10)):

    random.seed(0)
    np.random.seed(0)

    df = pd.read_csv(csv_path)

    samples = []

    patients = {}

    # Find each patients start of onset
    for kidx, pid in enumerate(df['PID'].unique()):

        print("(%.2f%%): %s" % (100*kidx/len(df['PID'].unique()), pid))

        patients[pid] = {'onset': None,
                         'positive': [],
                         'negative': []
                         }

        df_ep = df[df['PID'] == pid]
        episode = df_ep.values

        print("(%d): episode length - %s" % (pid, episode.shape[0]))

        for n in range(0, SAMPLES_PER_EPISODE):

            start = np.random.randint(0, len(episode) - int(seqlen)/2)
            sample = episode[start:start+seqlen]
            if len(sample) < seqlen:
                pad = seqlen - len(sample)
                sample = np.pad(sample, ((0, pad), (0, 0)))
            samples.append(sample)

    pyX = np.array(samples)
    print('pyX.npy: %s' % str(pyX.shape))
    print('mean = %.16f' % np.mean(pyX[:, :, 2:]))
    print('std  = %.16f' % np.std(pyX[:, :, 2:]))
    np.save('data/pyX.npy', pyX)


if __name__ == '__main__':
    plac.call(main)
