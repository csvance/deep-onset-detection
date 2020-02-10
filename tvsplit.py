import plac
import pandas as pd
import numpy as np
import random
import itertools
from scipy.signal import resample

# ECG Sampling Frequency
HZ = 200

# ECG Length (seconds)
LENGTH = 180

# Number of samples per episode
EP_SAMPLES = 36001

# Length of onset
ONSET = HZ * 10

# Onset Stride Size
STRIDE = 5

# Maximum samples to resample by
RESAMPLE = HZ


@plac.annotations(
    csv_path=('Path to the csv file', 'option', 'i', str),
    seqlen=('Length of the sequence to generate', 'option', 'l', int),
    tvsplit=('Training percentage', 'option', 't', float)
)
def main(csv_path: str = 'data/train_full_seq.csv',
         seqlen: int = HZ*10,
         tvsplit: float = 1.0):

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

        for sidx, sample in enumerate(episode):
            if sample[1] == 1.0:
                patients[pid]['onset'] = sidx
                print("(%d): onset at %d" % (pid, sidx))
                break

        if patients[pid]['onset'] is None:
            print("(%d): no onset." % pid)
            continue

        print("(%d): onset length - %d" % (pid, -1 + episode.shape[0] - patients[pid]['onset']))

        # Positive Sampling
        stop_idx = patients[pid]['onset'] + 1
        start_idx = stop_idx - seqlen

        if start_idx < 0:
            start_idx = 0
            stop_idx = seqlen

        while stop_idx < episode.shape[0]:

            resample_max = min(RESAMPLE, start_idx) if random.random() <= 0.5 else 0
            resample_count = np.random.randint(-resample_max, resample_max) if resample_max != 0 else 0

            sequence = episode[start_idx+resample_count:stop_idx, 1:]

            if len(sequence) != seqlen:
                sequence = resample(sequence, seqlen)

                x = sequence[:, 0]

                sequence[:, 0] = np.piecewise(x, [x <= 0.5, x > 0.5], [0, 1])

            pos_samples = np.sum(sequence[:, 0])
            pos_proportion = pos_samples / seqlen

            if pos_samples >= ONSET:
                break

            patients[pid]['positive'].append([sequence, pos_proportion])

            start_idx += STRIDE
            stop_idx += STRIDE

        print("(%d): %d positive samples." % (pid, len(patients[pid]['positive'])))

        potential_samples = patients[pid]['onset'] - seqlen
        if potential_samples < 1:
            continue

        # Negative
        generate_samples = min(potential_samples, len(patients[pid]['positive']))

        start_idxs = random.sample(range(potential_samples),
                                   generate_samples)

        for start_idx in start_idxs:
            stop_idx = start_idx + seqlen

            resample_max = min(RESAMPLE, start_idx) if random.random() <= 0.5 else 0
            resample_count = np.random.randint(-resample_max, resample_max) if resample_max != 0 else 0

            sequence = episode[start_idx+resample_count:stop_idx, 1:]

            if len(sequence) != seqlen:
                sequence = resample(sequence, seqlen)

                x = sequence[:, 0]

                sequence[:, 0] = np.piecewise(x, [x <= 0.5, x > 0.5], [0, 1])

            patients[pid]['negative'].append([sequence, 0])

        if len(patients[pid]['negative']) < len(patients[pid]['positive']):
            patients[pid]['negative'] = patients[pid]['negative'][:len(patients[pid]['positive'])]

        print("(%d): %d negative samples." % (pid, len(patients[pid]['negative'])))

        # Maintain balance between positive and negative samples
        min_samples = min(len(patients[pid]['negative']),
                          len(patients[pid]['positive']))

        patient = []

        patient.extend(patients[pid]['negative'][:min_samples])
        patient.extend(patients[pid]['positive'][:min_samples])

        print("(%d): %d samples" % (pid, len(patient)))

        samples.append(patient)

        # Free up memory
        del patients[pid]

    # Random patient shuffle
    random.shuffle(samples)

    split_point = int(np.round(len(samples) * tvsplit))

    train = list(itertools.chain(*samples[:split_point]))
    random.shuffle(train)

    train_samples = [i[0] for i in train]
    train_props = [i[1] for i in train]

    train = np.array(train_samples)
    train_props = np.array(train_props)
    train_props = np.reshape(train_props, (train_props.shape[0], 1))

    print('train.npy: %s' % str(train.shape))
    print('train_props.npy: %s' % str(train_props.shape))

    print('mean = %.16f' % np.mean(train[:, :, 1:]))
    print('std  = %.16f' % np.std(train[:, :, 1:]))

    np.save('data/train.npy', train)
    np.save('data/train_props.npy', train_props)

    if tvsplit < 1:
        val = list(itertools.chain(*samples[split_point:]))
        random.shuffle(val)

        val_samples = [i[0] for i in val]

        val = np.array(val_samples)
        print('val.npy: %s' % str(val.shape))
        np.save('data/val.npy', val)


if __name__ == '__main__':
    plac.call(main)
