"""
Plots using the posterior computed in an online manner with the model.
These plots are inspired by the plots presented in the BOCPD literature.
"""

import argparse
import bisect
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm

def myload(filename, start=None, end=None):
    """Reads a numpy array from a file from a txt or npy file. If start and end
     are provided, then the subset of columns [start, end) is retrieved."""
    if filename.endswith('npy'):
        ret = np.load(filename)
    else:
        ret = np.loadtxt(filename, ndmin=2)
    return ret[:, start:end]

def get_run_length_pmf_from_viterbi(vit_seq, start=None, end=None):
    assert vit_seq
    vit_seq = np.loadtxt(vit_seq).astype('int')
    maxdur = int(vit_seq[:,1].max())
    nobs = int(vit_seq[:,1].sum())
    ret = np.zeros((maxdur, nobs))
    idx = 0
    for hs, d in vit_seq:
        r = 0
        for i in xrange(d):
            ret[r][idx] = 1.0
            r += 1
            idx += 1
    assert idx == ret.shape[1]
    return ret[:, start:end]

def get_viterbi_sequence_from_vit_file(vit_file, start=None, end=None):
    vit_seq = np.loadtxt(vit_file).astype('int')
    sequence = []
    for hs, dur in vit_seq:
        sequence.append(np.ones((1, dur), dtype='int') * hs)
    sequence = np.hstack(sequence)
    return sequence[:, start:end]

def get_viterbi_sequence_from_statesm(statesm, start=None, end=None):
    statesm = myload(statesm, start, end)
    vit_filtering_seq = np.argmax(statesm, axis=0)
    return np.tile(vit_filtering_seq, (1, 1))

def plot_matrix(axis, mat, inverted=True):
    ret = axis.imshow(mat, cmap=plt.get_cmap('Greys'))
    axis.set_aspect('auto')
    if inverted:
        axis.invert_yaxis()
    return ret

def get_cdf_from_matrix(mat):
    rows, cols = mat.shape
    ret = np.copy(mat)
    for r in xrange(1, rows):
        ret[r, :] += ret[r - 1, :]
    return ret

def plot_time_series(axis, ts):
    if args.type == 'ecg':
        axis.plot(ts.T, color='purple')
    else:
        axis.plot(ts.T)

def plot_dashed_from_vit_file(axis, vit_file, run_length_flag, start=None,
        end=None):
    vit_seq = np.loadtxt(vit_file).astype('int')
    time_series = []
    for _, dur in vit_seq:
        if run_length_flag:
            time_series.extend(np.arange(dur))
        else:
            time_series.extend(np.arange(dur)[::-1])
    assert len(time_series) == vit_seq[:,1].sum()
    time_series = time_series[start:end]
    axis.plot(time_series, linestyle=(0, (5, 5)))

def plot_median_from_cdf(axis, cdfs):
    time_series = []
    for column in cdfs.T:
        time_series.append(bisect.bisect_left(column, 0.5))
    axis.plot(time_series, color='r')

def plot_mean_from_cdf(axis, cdfs):
    time_series = []
    for column in cdfs.T:
        time_series.append(bisect.bisect_left(column, 0.5))
    axis.plot(time_series, color='r')

def plot_max_from_pdf(axis, pdfs):
    axis.plot(np.argmax(pdfs, axis=0), color='r')

def plot_mean_from_pdf(axis, pdfs):
    time_series = []
    variance = []
    for column in pdfs.T:
        tmp = column * np.arange(np.size(column))
        var = column * (np.arange(np.size(column))**2)
        time_series.append(tmp.sum())
        variance.append(var.sum())
    time_series = np.array(time_series)
    variance = np.array(variance)
    variance = variance - (time_series**2)
    axis.plot(time_series, color='r')
    #axis.plot(time_series + 2*np.sqrt(variance), color='g')
    #axis.plot(time_series - 2*np.sqrt(variance), color='g')


SLEEP_STATES = ['Wake', 'NREM', 'REM']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obs', help="Observations file as a function of "
            "time (njointsXT dims)")
    parser.add_argument('--run_length', help="Run length pmf as a function of "
            "time (SXT dims)")
    parser.add_argument('--duration', help="Duration pmf as a function of "
            "time (DXT dims)")
    parser.add_argument('--statesm', help="States marginal pmf as a function"
        " of time (KXT dims)")
    parser.add_argument('--vitseq', help="viterbi sequence of the model")
    parser.add_argument('--gtvitseq', help="Ground truth Viterbi sequence")
    parser.add_argument('--mindur', type=int, default=0,
            help="Offsets the duration plot by the given quantity")
    parser.add_argument('--maxrunlength', type=int,
            help="Maximum runlength value that will be plotted")
    parser.add_argument('--maxduration', type=int,
            help="Maximum duration or residualtime  value that will be plotted")
    parser.add_argument('--start', type=int, help="Starting observation index"
            "(inclusive)")
    parser.add_argument('--end', type=int, help="Ending observation index"
            "(exclusive)")
    parser.add_argument('--filename', help='Filename to save the plot')
    parser.add_argument('--type', default='synth', help="Type of experiment")
    parser.add_argument('--argmax', action='store_true',
            help="Plot the argmax of the CDFs")
    parser.add_argument('--median', action='store_true',
            help="Plot the median of the CDFs")
    parser.add_argument('--mean', action='store_true',
            help="Plot the mean of the CDFs")
    args = parser.parse_args()
    plt.rc('text', usetex=True)
    fig, axis = plt.subplots(4, 1, sharex=True,
            gridspec_kw = {'height_ratios':[2, 1, 2, 2]})
    if args.obs:
        caxis = axis[0]
        obs = myload(args.obs, args.start, args.end)
        plot_time_series(caxis, obs)
        if args.type == 'sleep':
            caxis.set_ylabel('$\phi(\mathbf{y})$')
        elif args.type == 'ecg':
            caxis.set_ylabel('ECG[mV]')
        else:
            caxis.set_ylabel('Observations')
    list_m = []
    if args.vitseq:
        smoothing_vs = get_viterbi_sequence_from_vit_file(args.vitseq,
                args.start, args.end)
        list_m.append(smoothing_vs)
    if args.gtvitseq:
        gt_vs = get_viterbi_sequence_from_vit_file(args.gtvitseq,
                args.start, args.end)
        list_m.append(gt_vs)
    if args.statesm:
        filtering_vs = get_viterbi_sequence_from_statesm(args.statesm,
                args.start, args.end)
        list_m.append(filtering_vs)
    if list_m:
        caxis = axis[1]
        hidden_state_matrix = np.vstack(list_m)
        states = np.unique(hidden_state_matrix)
        nstates = len(states)
        cmap = ListedColormap(cm.get_cmap('Set1').colors[:nstates])
        norm = BoundaryNorm(np.arange(nstates + 1) - 0.5, cmap.N)
        ret = caxis.imshow(hidden_state_matrix, cmap=cmap, norm=norm)
        caxis.set_yticks([0, 1])
        caxis.set_yticklabels(['GT', 'MAP'])
        caxis.set_aspect('auto')
        labels=['S{}'.format(i) for i in range(nstates)]
        fontsize = 7
        if args.type == 'sleep':
            labels = SLEEP_STATES
            fontsize = 5
        colors = [ret.cmap(ret.norm(value)) for value in states]
        patches = [mpatches.Patch(color=colors[i],
            label=labels[i]) for i in range(nstates)]
        caxis.legend(handles=patches, prop={'size': fontsize},
                bbox_to_anchor=(1.03, 1))
    if args.run_length:
        caxis = axis[2]
        mat = myload(args.run_length, args.start, args.end)
        if args.argmax:
            plot_max_from_pdf(caxis, mat)
        if args.mean:
            plot_mean_from_pdf(caxis, mat)
        mat = get_cdf_from_matrix(mat)
        plot_matrix(caxis, mat)
        if args.median:
            plot_median_from_cdf(caxis, mat)
        if args.gtvitseq:
            plot_dashed_from_vit_file(caxis, args.gtvitseq, True, args.start,
                    args.end)
        if args.maxrunlength:
            caxis.set_ylim(top=args.maxrunlength)
        if args.type == 'sleep':
            caxis.set_ylabel('Run length', fontsize=7)
        else:
            caxis.set_ylabel('Run length')
    if args.duration:
        caxis = axis[3]
        duration = myload(args.duration, args.start, args.end)
        complete_duration_matrix = duration
        if args.mindur:
            padding = np.zeros((args.mindur - 1, duration.shape[1]))
            complete_duration_matrix = np.vstack((padding, duration))
        if args.argmax:
            plot_max_from_pdf(caxis, complete_duration_matrix)
        if args.mean:
            plot_mean_from_pdf(caxis, complete_duration_matrix)
        complete_duration_matrix = get_cdf_from_matrix(complete_duration_matrix)
        plot_matrix(caxis, complete_duration_matrix)
        if args.median:
            plot_median_from_cdf(caxis, complete_duration_matrix)
        if args.gtvitseq:
            plot_dashed_from_vit_file(caxis, args.gtvitseq, False, args.start,
                    args.end)
        if args.maxrunlength:
            caxis.set_ylim(top=args.maxduration)
        if args.type == 'sleep':
            caxis.set_ylabel('Residual time', fontsize=7)
        else:
            caxis.set_ylabel('Residual time')
    if args.type == 'sleep':
        plt.xlabel('EEG/EMG Epochs')
    else:
        plt.xlabel('Time steps')
    if args.type == 'sleep':
        fig.set_size_inches(8, 3)
    if args.filename:
        plt.savefig(args.filename, bbox_inches="tight")
    plt.show()

