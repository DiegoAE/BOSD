import os

import numpy as np

__location__ = os.path.realpath(os.path.join(os.getcwd(),
    os.path.dirname(__file__)))
obs_ecg = np.loadtxt(os.path.join(__location__, 'ecg.txt'))
gt_vit_seq_ecg = np.loadtxt(os.path.join(__location__, 'gt_vit_seq_ecg.txt'))
