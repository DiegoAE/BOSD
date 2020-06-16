import os

import numpy as np

__location__ = os.path.realpath(os.path.join(os.getcwd(),
    os.path.dirname(__file__)))
obs_eeg = np.loadtxt(os.path.join(__location__, 'sleep_features.txt.0'), ndmin=2)
gt_vit_seq_eeg = np.loadtxt(os.path.join(__location__,
        'sleep_gt_vit_labels.txt.0')).astype('int')
