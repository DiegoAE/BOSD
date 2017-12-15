#include <ForwardBackward.hpp>
#include <iostream>
#include <cmath>

using namespace arma;
using namespace std;

vec sampleFromDummyHMM(int n, int k, vec& states) {
    states = randi<vec>(n, distr_param(0, k - 1));
    cout << "Hidden states" << endl;
    cout << states << endl;
    vec B = randn<vec>(n) * 0.1;
    cout << "Obs" << endl;
    cout << states + B << endl;
    return states + B;
}


void viterbiPath(const imat& psi_d, const imat& psi_s, const mat& delta) {
    int nstates = delta.n_rows;
    int nobs = delta.n_cols;
    int curr_state = 0;
    int curr_obs_idx = nobs - 1;
    for(int i = 0; i < nstates; i++)
        if (delta(i, curr_obs_idx) > delta(curr_state, curr_obs_idx))
            curr_state = i;
    while(curr_obs_idx >= 0) {
        int d = psi_d(curr_state, curr_obs_idx);
        int next_state = psi_s(curr_state, curr_obs_idx);
        cout << "[" << curr_obs_idx - d + 1 << ", " << curr_obs_idx << "] -> " <<
                curr_state << endl;
        curr_obs_idx = curr_obs_idx - d;
        curr_state = next_state;
    }
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
    vec toy_states;
    vec toy_obs = sampleFromDummyHMM(nobs, nstates, toy_states);
    for(int i = 0; i < nstates; i++)
        for(int t = 0; t < nobs; t++)
            pdf(i, t, 0) = gaussianpdf(toy_obs(t), i, 0.1);
    mat alpha(nstates, nobs, fill::zeros);
    mat beta(nstates, nobs, fill::zeros);
    mat alpha_s(nstates, nobs, fill::zeros);
    mat beta_s(nstates, nobs, fill::zeros);
    FB(transition, pi, durations, pdf, alpha, beta, alpha_s, beta_s,
            min_duration, nobs);
//     cout << alpha << endl;
//     cout << beta << endl;
//     cout << "Sums columns" << endl;
//     cout << sum(alpha, 0) << endl;
//     cout << sum(beta, 0) << endl;
//     cout << "Sums rows" << endl;
//     cout << sum(alpha, 1) << endl;
//     cout << sum(beta, 1) << endl;
    imat psi_duration(nstates, nobs, fill::zeros);
    imat psi_state(nstates, nobs, fill::zeros);
    mat delta(nstates, nobs, fill::zeros);
    Viterbi(transition, pi, durations, pdf, delta, psi_duration, psi_state,
            min_duration, nobs);
    cout << delta.col(nobs - 1) << endl;
    viterbiPath(psi_duration, psi_state, delta);

    // Debug
    vec debug = {3, 1, 2, 3, 0, 1, 1, 0, 2, 1};
    double real_acum = 1.0;
    double debug_acum = 1.0;
    for(int t = 0; t < nobs; t++) {
        real_acum *= gaussianpdf(toy_obs(t) - toy_states(t), 0, 0.1) * 0.25;
        debug_acum *= gaussianpdf(toy_obs(t) - debug(t), 0, 0.1) * 0.25;
    }
    cout << "Real: " << real_acum << " Debug: " << debug_acum << endl;
    // TODO: Test the alpha and beta recursions with this example.
    // TODO: Test with a duration greater than 1.
    // TODO: implement logs in the Viterbi algorithm.
    return 0;
}