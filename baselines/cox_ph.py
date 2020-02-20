import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter

from data.ecg import obs_ecg, gt_vit_seq_ecg

MAX_DUR = 180
K = 10
FRAC = 0.01

def get_pmf_from_survival(survival_f):
    pmf = survival_f.copy()
    for i in range(survival_f.shape[0] - 1):
        pmf[i, :] -= pmf[i + 1, :]
    sums = np.sum(pmf, axis=0)

    #plt.plot(np.arange(MAX_DUR), pmf)
    #plt.show()
    assert (sums > 0.95).all(), survival_f[0]
    return pmf

def loglikelihood(hazards, cum_hazards):
    """ Refer to https://stats.stackexchange.com/questions/417303/
    what-is-the-likelihood-for-this-process"""
    assert hazards.shape == cum_hazards.shape
    lls = np.log(hazards) - cum_hazards
    return lls

def get_hazard_from_cum_hazard(cum_hazard):
    """
        Refer to the Discrete survival models section in lifelines.
    """
    hazard = 1 - np.exp(cum_hazard[:-1,:] - cum_hazard[1:,:])
    return hazard

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
    #cph.print_summary()
    return cph

def cross_validation(df):
    df_test = df.sample(frac=FRAC)
    df_train = df.loc[~df.index.isin(df_test.index)]
    cph = fit_cph(df_train)
    times_to_predict = np.arange(MAX_DUR)
    survival_f = cph.predict_survival_function(df_test,
            times=times_to_predict)
    survival_f = survival_f.values
    pmf = get_pmf_from_survival(survival_f)
    ll = 0
    gt_survival_times = df_test['survival_time'].values
    for i in range(len(gt_survival_times)):
        ll += np.log(pmf[gt_survival_times[i], i])
    return ll


if __name__ == '__main__':
    np.random.seed(10)
    for nobs in [5, 10, 20, 40]:
        print('nobs {} ========'.format(nobs))
        ecg_pd = get_ecg_pd(nobs)
        s0_pd = ecg_pd.loc[~ecg_pd['state'].astype(bool)]
        s1_pd = ecg_pd.loc[ecg_pd['state'].astype(bool)]

        state_pds = [s0_pd, s1_pd]
        results = []
        for _ in range(K):
            l = []
            for s_pd in state_pds:
                l.append(cross_validation(s_pd))
            results.append(l)
        results = np.array(results)
        print('Means', np.mean(results, axis=0))
        print('Stds', np.std(results, axis=0))
