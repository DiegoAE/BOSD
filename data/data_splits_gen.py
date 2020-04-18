import argparse
import numpy as np


def data_splits_gen(input_obs, input_viterbi_labels, splits):
    assert splits > 0
    nsegments = input_viterbi_labels.shape[0]
    sizes = [nsegments // splits] * splits
    for i in range(nsegments % splits):
        sizes[i] += 1
    assert sum(sizes) == nsegments
    vit_idx = 0
    obs_idx = 0
    for size in sizes:
        vit = input_viterbi_labels[vit_idx:vit_idx + size]
        nobs = vit[:,1].sum()
        obs = input_obs[obs_idx:obs_idx + nobs]
        vit_idx += size
        obs_idx += nobs
        yield obs, vit
    assert vit_idx == nsegments and obs_idx == input_obs.shape[0]

def ecg_data_splits_gen():
    from data.ecg import obs_ecg, gt_vit_seq_ecg

    return data_splits_gen(obs_ecg, gt_vit_seq_ecg, 20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splits the given observations'
            ' and viterbi labels in a certain number of files consistently.')
    parser.add_argument('input_obs')
    parser.add_argument('input_vit')
    parser.add_argument('nsplits', type=int)
    parser.add_argument('--output_obs_split',
            help='Path to save an observation split')
    parser.add_argument('--output_vit_split',
            help='Path to save viterbi labels split')
    args = parser.parse_args()
    obs = np.loadtxt(args.input_obs)
    vit = np.loadtxt(args.input_vit).astype('int')
    s = 0
    for o, v in data_splits_gen(obs, vit, args.nsplits):
        print('Split {} has {} observations and {} segments'.format(
            s, o.shape[0], v.shape[0]))
        if args.output_obs_split:

            # Note the reshaping such that it's compatible.
            np.savetxt(args.output_obs_split + '.' + str(s), o.reshape((1,-1)))
        if args.output_vit_split:
            np.savetxt(args.output_vit_split + '.' + str(s), v, fmt='%i')
        s += 1
