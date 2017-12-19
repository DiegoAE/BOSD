#include <ForwardBackward.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace arma;
using namespace std;


int sampleFromCategorical(rowvec pmf) {
    rowvec prefixsum(pmf);
    for(int i = 1; i < pmf.n_elem; i++)
        prefixsum(i) += prefixsum(i - 1);
    return lower_bound(prefixsum.begin(), prefixsum.end(), randu()) -
            prefixsum.begin();
}

class DummyHSMM {
    public:
        DummyHSMM(mat transition, vec pi, vec duration, int min_duration) {
            transition_ = transition;
            pi_ = pi;
            duration_ = duration;
            min_duration_ = min_duration;
        }

        int getNumberOfStates() {
            return transition_.n_rows;
        }

        rowvec sampleFromState(int state, int len) {
            return randn<vec>(len) * 0.1 + state;
        }

        mat sampleSegments(int nsegments, ivec& hiddenStates,
                ivec& hiddenDurations) {
            assert(nsegments >= 1);

            // Generating states sequence.
            ivec states(nsegments);
            states(0) = sampleFromCategorical(pi_.t());
            for(int i = 1; i < nsegments; i++) {
                rowvec nstateDist = transition_.row(states(i - 1));
                states(i) = sampleFromCategorical(nstateDist);
            }

            // Generating durations from states.
            int sampleSequenceLength = 0;
            ivec durations(nsegments);
            for(int i = 0; i < nsegments; i++) {
                int currentDuration = sampleFromCategorical(
                        duration_.row(states(i))) + min_duration_;
                durations(i) = currentDuration;
                sampleSequenceLength += currentDuration;
            }

            // Generating samples
            mat samples(1, sampleSequenceLength, fill::zeros);
            int idx = 0;
            for(int i = 0; i < nsegments; i++) {
                rowvec currSample = sampleFromState(states(i), durations(i));
                samples.cols(idx, idx + durations(i) - 1) = currSample;
                idx += durations(i);
            }
            hiddenStates = states;
            hiddenDurations = durations;
            return samples;
        }

        mat transition_;
        vec pi_;
        vec duration_;
        int min_duration_;
};

void viterbiPath(const imat& psi_d, const imat& psi_s, const mat& delta,
        ivec& hiddenStates, ivec& hiddenDurations) {
    int nstates = delta.n_rows;
    int nobs = delta.n_cols;
    int curr_state = 0;
    int curr_obs_idx = nobs - 1;
    for(int i = 0; i < nstates; i++)
        if (delta(i, curr_obs_idx) > delta(curr_state, curr_obs_idx))
            curr_state = i;
    vector<int> statesSeq, durationSeq;
    while(curr_obs_idx >= 0) {
        int d = psi_d(curr_state, curr_obs_idx);
        int next_state = psi_s(curr_state, curr_obs_idx);

        // Making sure that the Viterbi algorithm ran correctly.
        assert(d >= 1);

        // current segment: [curr_obs_idx - d + 1, curr_obs_idx] -> curr_state.
        statesSeq.push_back(curr_state);
        durationSeq.push_back(d);
        curr_obs_idx = curr_obs_idx - d;
        curr_state = next_state;
    }
    reverse(statesSeq.begin(), statesSeq.end());
    reverse(durationSeq.begin(), durationSeq.end());
    hiddenStates = conv_to<ivec>::from(statesSeq);
    hiddenDurations = conv_to<ivec>::from(durationSeq);
}

double gaussianpdf(double x, double mu, double sigma) {
    double ret = exp(((x - mu)*(x - mu)) / (-2*sigma*sigma));
    ret = ret / (sqrt(2 * M_PI) * sigma);
    return ret;
}

int main() {
    int nobs = 10;
    int nstates = 4;
    int ndurations = 1;
    int min_duration = 1;
    mat transition(nstates, nstates, fill::eye);
    transition.fill(1.0/nstates);
    vec pi(nstates, fill::eye);
    pi.fill(1.0/nstates);
    mat durations(nstates, ndurations, fill::zeros);
    durations.col(0) = ones<vec>(nstates);
    cube pdf(nstates, nobs, ndurations, fill::zeros);

    DummyHSMM dhsmm(transition, pi, durations, min_duration);
    ivec hiddenStates, hiddenDurations;
    mat samples = dhsmm.sampleSegments(nobs, hiddenStates, hiddenDurations);

    cout << "Generated samples" << endl;
    cout << samples << endl;
    cout << "Generated states and durations" << endl;
    cout << join_horiz(hiddenStates, hiddenDurations) << endl;

    for(int i = 0; i < nstates; i++)
        for(int t = 0; t < nobs; t++)
            pdf(i, t, 0) = gaussianpdf(samples(0, t), i, 0.1);

    mat alpha(nstates, nobs, fill::zeros);
    mat beta(nstates, nobs, fill::zeros);
    mat alpha_s(nstates, nobs, fill::zeros);
    mat beta_s(nstates, nobs, fill::zeros);
    FB(transition, pi, durations, pdf, alpha, beta, alpha_s, beta_s,
            min_duration, nobs);
    cout << "Alpha" << endl;
    cout << alpha << endl;
    cout << "Beta" << endl;
    cout << beta << endl;
    cout << "Alpha normalized" << endl;
    mat alpha_normalized(alpha);
    for(int i = 0; i < alpha.n_rows; i++)
        alpha_normalized.row(i) /= sum(alpha, 0);
    cout << alpha_normalized << endl;
    cout << "Sums columns" << endl;
    cout << sum(alpha, 0) << endl;
    cout << sum(beta, 0) << endl;
    cout << "Sums rows" << endl;
    cout << sum(alpha, 1) << endl;
    cout << sum(beta, 1) << endl;
    imat psi_duration(nstates, nobs, fill::zeros);
    imat psi_state(nstates, nobs, fill::zeros);
    mat delta(nstates, nobs, fill::zeros);
    Viterbi(transition, pi, durations, pdf, delta, psi_duration, psi_state,
            min_duration, nobs);
    cout << "Delta last column" << endl;
    cout << delta.col(nobs - 1) << endl;
    ivec viterbiStates, viterbiDurations;
    viterbiPath(psi_duration, psi_state, delta, viterbiStates,
            viterbiDurations);

    cout << "Viterbi states and durations" << endl;
    cout << join_horiz(viterbiStates, viterbiDurations) << endl;

    // Debug
    double real_acum = 1.0;
    double debug_acum = 1.0;
    int differences = 0;
    for(int t = 0; t < nobs; t++) {
        real_acum *= gaussianpdf(samples(0, t) -
                hiddenStates(t), 0, 0.1) * 0.25;
        debug_acum *= gaussianpdf(samples(0, t) -
                viterbiStates(t), 0, 0.1) * 0.25;
        if (hiddenStates(t) != viterbiStates(t))
            differences++;
    }
    cout << "Real seq: " << real_acum << " Viterbi seq " << debug_acum <<
            " Differences: " << differences << endl;

    // TODO: Test the beta recursions with this example.
    // TODO: Test with a duration greater than 1.
    return 0;
}