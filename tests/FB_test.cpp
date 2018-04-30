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

    logsFB(log(transition), log(pi), log(duration), logpdf, full_labels,
            alpha, beta, alpha_s, beta_s, beta_s_0, eta, min_duration, nobs);
    double llikelihood = logsumexp(alpha.col(nobs - 1));
	BOOST_CHECK(fabs(llikelihood - expected_llikelihood_fulllabels) < EPSILON);

    // Testing with sparse labels.
    logsFB(log(transition), log(pi), log(duration), logpdf, full_labels,
            alpha, beta, alpha_s, beta_s, beta_s_0, eta, min_duration, nobs);
    llikelihood = logsumexp(alpha.col(nobs - 1));
    cube posterior_eta = exp(eta - llikelihood);
    for(auto p : seq_unobserved_segments) {
        int sum_starting = 0;
        int sum_ending = 0;
        int start_seg = p.first;
        int end_seg = p.second;
        for(int i = 0; i < nstates; i++)
            for(int d = 0; d < ndurations; d++) {
                sum_starting += posterior_eta(i, d,
                        start_seg + min_duration + d - 1);
                sum_ending += posterior_eta(i, d, end_seg);
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
            BOOST_CHECK(fabs(posterior_eta(hs, d - min_duration,
                            current_idx - 1) - 1) < EPSILON);
    }

    // Checking that the expected number of segments in the observation
    // sequence is equal to the actual value. This is expected because
    // the actual parameters are used for inference.
    BOOST_CHECK(fabs(accu(posterior_eta) - nSampledSegments) < EPSILON);
}



