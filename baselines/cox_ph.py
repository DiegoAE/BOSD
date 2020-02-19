import argparse

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from data.ecg import obs_ecg, gt_vit_seq_ecg


def get_covariates_dict_from_list(covs_list):
    d = {}
    for i in range(len(covs_list[0])):
        l = []
        for covs in covs_list:
            l.append(covs[i])
        d['c{}'.format(i)] = l
    return d

def get_ecg_pd(nobs):
    """ Get a pandas data frame from the ecg data. nobs denotes the number of
        observations that will be feed into the predictive model."""
    assert nobs > 0
    survival_time = []
    state = []
    ecg = []
    acum = 0
    for s, d in gt_vit_seq_ecg:
        ecg.append(obs_ecg[acum: acum + nobs]) 
        state.append(s)
        st = d - nobs
        assert(st >= 0)
        survival_time.append(st)
        acum += d
    finished = [1] * len(state)  # All segments are complete. No censoring.
    ecg_dict = {'survival_time': survival_time, 'state': state,
            'finished': finished}
    covs_dict = get_covariates_dict_from_list(ecg)
    ecg_dict.update(covs_dict)
    return pd.DataFrame(ecg_dict)

def fit_cph(data):
    cph = CoxPHFitter()
    cph.fit(data, duration_col='survival_time', event_col='finished',
            strata=['state'])
    cph.print_summary()


if __name__ == '__main__':
    cph = CoxPHFitter()
    for nobs in [5, 10, 20, 40]:
        print(nobs)
        ecg_pd = get_ecg_pd(nobs)
        s0_pd = ecg_pd.loc[~ecg_pd['state'].astype(bool)]
        s1_pd = ecg_pd.loc[ecg_pd['state'].astype(bool)]
        fit_cph(s0_pd)
        fit_cph(s1_pd)
