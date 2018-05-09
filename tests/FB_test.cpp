#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ForwardBackward
#include <boost/test/unit_test.hpp>
#include <HSMM.hpp>
#include <ForwardBackward.hpp>
#include <memory>
#include <exception>
#include <set>

#define EPSILON 1e-6

using namespace arma;
using namespace hsmm;
using namespace std;

mat transition = {{0.0, 0.1, 0.4, 0.5},
                  {0.3, 0.0, 0.6, 0.1},
                  {0.2, 0.2, 0.0, 0.6},
                  {0.4, 0.4, 0.2, 0.0}};

mat duration = {{0.0, 0.1, 0.4, 0.5},
                {0.3, 0.0, 0.6, 0.1},
                {0.2, 0.2, 0.0, 0.6},
                {0.4, 0.4, 0.2, 0.0}};

int min_duration = 4;

vec pi = {0.25, 0.25, 0.25, 0.25};

int ndurations = duration.n_cols;

int nstates = duration.n_rows;

// Emission parameters.
vec means = {0, 5, 10, 15};
vec std_devs =  {0.5, 1.0, 0.1, 2.0};

BOOST_AUTO_TEST_CASE( ForwardBackwardWithLabels ) {
    shared_ptr<AbstractEmission> ptr_emission(new DummyGaussianEmission(
            means, std_devs));
    HSMM dhsmm(ptr_emission, transition, pi, duration, min_duration);
    ivec hiddenStates, hiddenDurations;
    int nSampledSegments = 50;
    mat samples = dhsmm.sampleSegments(nSampledSegments, hiddenStates,
            hiddenDurations);
    int nobs = samples.n_cols;

    // Output parameters of the Forward-Backward algorithm.
    mat alpha(nstates, nobs, fill::zeros);
    mat beta(nstates, nobs, fill::zeros);
    mat alpha_s(nstates, nobs, fill::zeros);
    mat beta_s(nstates, nobs, fill::zeros);
    vec beta_s_0(nstates, fill::zeros);
    cube eta(nstates, ndurations, nobs, fill::zeros);
    cube zeta(nstates, nstates, nobs - 1, fill::zeros);

    cube logpdf = dhsmm.computeEmissionsLogLikelihood(samples);
    Labels full_labels;
    Labels sparse_labels;
    set<int> sparse_segment_ids = {10, 30, 40, 45};

    // Required for generating the runs of unobserved segments.
    vector<pair<int, int>> seq_unobserved_segments;
    int starting_unobserved_segment_idx = 0;

    int current_idx = 0;
    double expected_llikelihood_fulllabels = 0;
    for(int i = 0; i < nSampledSegments; i++) {
        int hs = hiddenStates(i);
        int d = hiddenDurations(i);
        expected_llikelihood_fulllabels += logpdf(hs, current_idx,
                d - min_duration);
        int starting_idx = current_idx;
        current_idx += d;
        int ending_idx = current_idx - 1;
        full_labels.setLabel(ending_idx, d, hs);
        if (sparse_segment_ids.find(i) != sparse_segment_ids.end()) {
            sparse_labels.setLabel(ending_idx, d, hs);
            seq_unobserved_segments.push_back(make_pair(
                    starting_unobserved_segment_idx, starting_idx - 1));
            starting_unobserved_segment_idx = ending_idx + 1;
        }
    }
    seq_unobserved_segments.push_back(make_pair(starting_unobserved_segment_idx,
                current_idx - 1));

    // Testing with full labels.
    logsFB(log(transition), log(pi), log(duration), logpdf, full_labels,
            alpha, beta, alpha_s, beta_s, beta_s_0, eta, zeta, min_duration,
            nobs);
    eta = exp(eta);
    zeta = exp(zeta);
    double llikelihood = logsumexp(alpha.col(nobs - 1));
	BOOST_CHECK(fabs(llikelihood - expected_llikelihood_fulllabels) < EPSILON);
    current_idx = 0;
    for(int i = 0; i < nSampledSegments - 1; i++) {
        int hs = hiddenStates(i);
        int next_hs = hiddenStates(i + 1);
        int d = hiddenDurations(i);
        current_idx += d;

        // Checking that the observed transitions have probability one and the
        // rest have zero probability.
        for(int j = 0; j < nstates; j++)
            for(int k = 0; k < nstates; k++)
                if (j == hs && k == next_hs)
                    BOOST_CHECK(fabs(zeta(j, k, current_idx - 1) - 1.0) < EPSILON);
                else
                    BOOST_CHECK(fabs(zeta(j, k, current_idx - 1) - 0.0) < EPSILON);
    }

    // Testing with sparse labels.
    logsFB(log(transition), log(pi), log(duration), logpdf, sparse_labels,
            alpha, beta, alpha_s, beta_s, beta_s_0, eta, zeta, min_duration,
            nobs);
    eta = exp(eta);
    for(auto p : seq_unobserved_segments) {
        double sum_starting = 0;
        double sum_ending = 0;
        int start_seg = p.first;
        int end_seg = p.second;
        for(int i = 0; i < nstates; i++)
            for(int d = 0; d < ndurations; d++) {
                sum_starting += eta(i, d, start_seg + min_duration + d - 1);
                sum_ending += eta(i, d, end_seg);
            }

        // There must be a segment which explains the starting part of an
        // unobserved run.
        BOOST_CHECK(fabs(sum_starting - 1.0) < EPSILON);

        // The same applies for the ending part.
        BOOST_CHECK(fabs(sum_ending - 1.0) < EPSILON);
    }

    current_idx = 0;
    for(int i = 0; i < nSampledSegments; i++) {
        int hs = hiddenStates(i);
        int d = hiddenDurations(i);
        current_idx += d;

        // Checking that the provided labels apper as ones in eta.
        if (sparse_segment_ids.find(i) != sparse_segment_ids.end())
            BOOST_CHECK(fabs(eta(hs, d - min_duration, current_idx - 1) - 1) <
                    EPSILON);
    }

    // Checking that the expected number of segments in the observation
    // sequence is equal to the actual value. This is expected because
    // the actual parameters are used for inference.
    BOOST_CHECK(fabs(accu(eta) - nSampledSegments) < EPSILON);
}

BOOST_AUTO_TEST_CASE( LogSumExp ) {
    int n = 10;
    vec ones_v = ones<vec>(n);
    double sum = exp(logsumexp(log(ones_v)));
    BOOST_CHECK(fabs(sum - n) < EPSILON);
    int nzeros = 3;
    ones_v.subvec(0, nzeros - 1).fill(0.0);
    sum = exp(logsumexp(log(ones_v)));
    BOOST_CHECK(fabs(sum - (n-nzeros)) < EPSILON);
}

