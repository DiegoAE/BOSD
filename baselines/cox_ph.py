""" Survival regression with Cox's proportional hazard model. """

import argparse
import logging
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter


def get_pmf_from_survival(survival_f):
    pmf = survival_f.copy()
    for i in range(survival_f.shape[0] - 1):
        pmf[i] -= pmf[i + 1]
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
    m = np.array([c.flatten() for c in covs_list])
    d = {}
    for i, dim in enumerate(m.T):
        d['c{}'.format(i)] = dim
    return d

def get_state_sequence_and_residual_time_from_vit(vit):
    states = []
    residual_times = []
    for hs, dur in vit:
        states.extend([hs] * dur)

        # Note that we're shifting by one the residual times here.
        residual_times.extend(np.arange(dur)[::-1] + 1)
    assert len(states) == len(residual_times) and len(states) == vit[:,1].sum()
    return states, residual_times

def get_pd(horizon, lobs, lvit, ntest_obs=None):
    """ Get a pandas data frame from input data (lobs, lvit). horizon denotes
        the number of observations that will be feed into the predictive model.
    """
    assert horizon > 0
    survival_times = []
    covariates = []
    for obs, vit in zip(lobs, lvit):
        dim, nobs = obs.shape
        _, residual_times = get_state_sequence_and_residual_time_from_vit(vit)
        assert len(residual_times) == nobs
        if ntest_obs is not None:
            nobs = ntest_obs
        for t in range(nobs - horizon + 1):
            covariates.append(obs[:, t: t + horizon])
            survival_time = residual_times[t + horizon - 1]
            assert survival_time >= 0
            survival_times.append(survival_time)
    complete = [1] * len(survival_times)  # All segments are complete. No censoring.
    data_dict = {'survival_time': survival_times, 'complete': complete}
    covariates_dict = get_covariates_dict_from_list(covariates)
    data_dict.update(covariates_dict)
    return pd.DataFrame(data_dict)

def fit_cph(data):
    return cph

def fit_km(training_data, test_data, times_to_predict):
    kmf = KaplanMeierFitter()
    kmf.fit(training_data['survival_time'])
    survival_f = kmf.survival_function_at_times(times_to_predict)
    pmf = get_pmf_from_survival(
            kmf.survival_function_at_times(times_to_predict).values)
    ll = 0
    gt = test_data['survival_time'].values
    for l in gt:
        ll += np.log(pmf[l])
    one_hot_gt = np.eye(len(pmf))[gt]
    #upmf = np.array([1./len(pmf)] * len(pmf))
    return ll, brier_score(pmf, one_hot_gt), np.log(1./len(pmf)) * len(gt)

def fit_cph(train_pd, test_pd, times_to_predict):
    cph = CoxPHFitter()
    cph.fit(train_pd, duration_col='survival_time', event_col='complete')
    # cph.print_summary()
    survival_f = cph.predict_survival_function(test_pd,
            times=times_to_predict)
    survival_f = survival_f.values
    pmfs = get_pmf_from_survival(survival_f)
    ll = 0
    gt = test_pd['survival_time'].values
    for pmf,l in zip(pmfs.T, gt):
        ll += np.log(pmf[l])
    one_hot_gt = np.eye(len(pmf))[gt]
    return ll, brier_score(pmfs.T, one_hot_gt), np.log(1./len(pmf)) * len(gt)

def brier_score(pmf, one_hot_gt):
    return np.sum(np.square(pmf - one_hot_gt)) / len(one_hot_gt)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.random.seed(10)
    parser = argparse.ArgumentParser(description=__doc__)

    # Training arguments.
    parser.add_argument('horizon', type=int)
    parser.add_argument('training_obs')
    parser.add_argument('training_vit')
    parser.add_argument('nfiles', type=int)

    # Testing arguments. Note that a single file is assumed.
    parser.add_argument('testing_obs')
    parser.add_argument('testing_vit')

    parser.add_argument('--ntest_obs', type=int)

    args = parser.parse_args()
    logging.info('horizon: ' + str(args.horizon))

    lobs = []
    lvit = []
    for i in range(args.nfiles):
        obs = np.loadtxt(args.training_obs + '.' + str(i), ndmin=2)
        vit = np.loadtxt(args.training_vit + '.' + str(i)).astype('int')
        lobs.append(obs)
        lvit.append(vit)

    # Testing pandas data frame.
    test_obs = np.loadtxt(args.testing_obs, ndmin=2)
    test_vit = np.loadtxt(args.testing_vit).astype('int')
    test_pd = get_pd(args.horizon, [test_obs], [test_vit], args.ntest_obs)
    print('Number of test predictions:',  len(test_pd['survival_time']))

    lls_km = []
    bss_km = []
    rgs = []

    lls_cph = []
    bss_cph = []
    rgs_cph = []
    for leave_one_out in range(args.nfiles):
        lobs_copy = list(lobs)
        lvit_copy = list(lvit)
        lobs_copy.pop(leave_one_out)
        lvit_copy.pop(leave_one_out)

        # Note the +1 since the min residual value is 1.
        times_to_predict = np.arange(181)

        # Training pandas data frame.
        train_pd = get_pd(args.horizon, lobs_copy, lvit_copy)

        # Fitting KM estimator.
        ll_km, bs_km, rg = fit_km(train_pd, test_pd, times_to_predict)
        lls_km.append(ll_km)
        bss_km.append(bs_km)
        rgs.append(rg)

        # Fitting Cox's proportional hazards model.
        ll_cph, bs_cph, rg = fit_cph(train_pd, test_pd, times_to_predict)
        lls_cph.append(ll_cph)
        bss_cph.append(bs_cph)
        rgs_cph.append(rg)

        print('Iter {}. KM ll: {}. CPH ll: {}'.format(leave_one_out, ll_km, ll_cph))

    print('KM.\n\tll: {} +- {}\n\tbs: {} +- {}\n\trg {} +- {}'.format(
        np.mean(lls_km), np.std(lls_km),
        np.mean(bss_km), np.std(bss_km),
        np.mean(rgs), np.std(rgs)))
    print('CPH.\n\tll: {} +- {}\n\tbs: {} +- {}\n\trg {} +- {}'.format(
        np.mean(lls_cph), np.std(lls_cph),
        np.mean(bss_cph), np.std(bss_cph),
        np.mean(rgs_cph), np.std(rgs_cph)))
