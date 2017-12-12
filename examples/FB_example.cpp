#include <ForwardBackward.hpp>
#include <iostream>
#include <cmath>

using namespace arma;
using namespace std;

vec sampleFromDummyHMM(int n, int k) {
    vec A = randi<vec>(n, distr_param(0, k - 1));
    // cout << A << endl;
    vec B = randn<vec>(n);
    return A + B;
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
    vec toy_obs = sampleFromDummyHMM(nobs, nstates);
    for(int i = 0; i < nstates; i++)
        for(int t = 0; t < nobs; t++)
            pdf(i, t, 0) = gaussianpdf(toy_obs(t), i, 1.0);
    mat alpha(nstates, nobs, fill::zeros);
    mat beta(nstates, nobs, fill::zeros);
    mat alpha_s(nstates, nobs, fill::zeros);
    mat beta_s(nstates, nobs, fill::zeros);
    FB(transition, pi, durations, pdf, alpha, beta, alpha_s, beta_s,
            min_duration, nobs);
    cout << alpha << endl;
    cout << beta << endl;
    cout << "Sums columns" << endl;
    cout << sum(alpha, 0) << endl;
    cout << sum(beta, 0) << endl;
    cout << "Sums rows" << endl;
    cout << sum(alpha, 1) << endl;
    cout << sum(beta, 1) << endl;
    return 0;
}