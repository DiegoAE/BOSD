import argparse
import numpy as np


def data_splits_gen(input_obs, input_viterbi_labels, splits):
    assert splits > 0
    nsegments = input_viterbi_labels.shape[0]
    sizes = [nsegments // splits] * splits
    for i in range(nsegments % splits):
        sizes[i] += 1
    assert sum(sizes) == nsegments
    idx = 0
    for size in sizes:
        yield input_obs[idx:idx + size], input_viterbi_labels[idx:idx + size]

def ecg_data_splits_gen():
    from data.ecg import obs_ecg, gt_vit_seq_ecg

    return data_splits_gen(obs_ecg, gt_vit_seq_ecg, 20)

if __name__ == '__main__':
    for x, y in ecg_data_splits_gen():
        print(x.shape, y.shape)
