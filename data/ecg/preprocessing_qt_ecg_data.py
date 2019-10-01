"""
Preprocesses ECG signal to be used within the HSMM framework.
It outputs two files:
    * The ecg observations.
    * The corresponding viterbi file generated using the marked events in the database.
"""

import matplotlib.pyplot as plt
import numpy as np
import wfdb

FROM = 0
TO = None
SUBJECT = 'sel100'
SIGNAL_ID = 1
PLOT_FLAG = False
OUTPUT_SIGNAL = 'ecg.txt'
OUTPUT_VIT = 'gt_vit_seq_ecg.txt'

def plot_realizations(signal, boundaries):
    for i in range(len(boundaries) - 1):
        left = boundaries[i]
        right = boundaries[i + 1]
        s = signal[left:right]
        plt.plot(np.linspace(0, 1, len(s)), s)
    plt.show()

def sanity_checks(symbols):
    for i in xrange(len(symbols)):
        c = symbols[i]
        if c == 'N' or c == 'p':
            l = symbols[i - 1]
            r = symbols[i + 1]
            assert l == '(' and r == ')'

def get_boundaries_for_systole_diastole_sequence(symbols, samples):
    assert len(symbols) == len(samples)
    ret = []
    last = None
    if PLOT_FLAG:
        fig, axis = plt.subplots(2, 1)
    for i in xrange(len(symbols)):
        event = symbols[i]
        if event == 'N':
            assert last == 't' or last is None, "last: {}. idx: {}.".format(
                    last, samples[i])
            if PLOT_FLAG and last:
                segment = signal[ret[-1]:samples[i]]
                axis[0].plot(np.linspace(0,1,len(segment)), segment)
            last = event
            ret.append(samples[i])
        elif event == 't' and symbols[i+1] == ')':
            assert last == 'N'
            if PLOT_FLAG:
                segment = signal[ret[-1]:samples[i+1]]
                axis[1].plot(np.linspace(0,1,len(segment)), segment)
            last = event
            ret.append(samples[i+1])
    if PLOT_FLAG:
        plt.show()
    return np.array(ret)


if __name__ == "__main__":
    record = wfdb.rdrecord(SUBJECT, pb_dir='qtdb', sampfrom=FROM,
            sampto=TO)
    annotation = wfdb.rdann(SUBJECT, 'pu', pb_dir='qtdb', sampfrom=FROM,
            sampto=TO)
    (signal, ann_samp, ann_sym, fs, ylabel,
            record_name) =  wfdb.plot.plot.get_wfdb_plot_items(
            record, annotation, True)
    signal = signal[:, SIGNAL_ID]
    ann_samp = ann_samp[SIGNAL_ID]
    ann_sym = ann_sym[SIGNAL_ID]
    N_events = []
    sanity_checks(ann_sym)
    for event, time_step in zip(ann_sym, ann_samp):
        if event == 'N':
            N_events.append(time_step)
    N_durs = []
    for i  in range(len(N_events) - 1):
        N_durs.append(N_events[i + 1] - N_events[i])
    N_durs = np.array(N_durs)
    print "Number of segments: {}".format(len(N_durs))
    if PLOT_FLAG:
        plot_realizations(signal, N_events)
        plt.hist(N_durs)
        plt.show()
    systole_diastole_boundaries = get_boundaries_for_systole_diastole_sequence(
            ann_sym, ann_samp)
    assert all(x<y for x,y in zip(systole_diastole_boundaries,
            systole_diastole_boundaries[1:]))
    vit = []
    for i in xrange(len(systole_diastole_boundaries) - 1):
        dur = systole_diastole_boundaries[i+1] - systole_diastole_boundaries[i]
        entry = [i % 2, dur]
        vit.append(entry)
    vit = np.array(vit, dtype='int')
    np.savetxt(OUTPUT_VIT, vit, fmt='%i')
    obs = signal[systole_diastole_boundaries[0]:systole_diastole_boundaries[-1]]
    obs = obs.reshape((1,-1))
    np.savetxt(OUTPUT_SIGNAL, obs)

