#include <armadillo>
#include <iostream>
#include <ForwardBackward.hpp>
#include <HSMM.hpp>
#include <memory>

using namespace arma;
using namespace hsmm;
using namespace std;

int main() {
    int ndurations = 4;
    int min_duration = 4;
    mat transition = {{0.0, 0.1, 0.4, 0.5},
                      {0.3, 0.0, 0.6, 0.1},
                      {0.2, 0.2, 0.0, 0.6},
                      {0.4, 0.4, 0.2, 0.0}};
    int nstates = transition.n_rows;
    vec pi(nstates, fill::eye);
    pi.fill(1.0/nstates);
    // mat durations(nstates, ndurations, fill::eye);
    mat durations =  {{0.0, 0.1, 0.4, 0.5},
                      {0.3, 0.0, 0.6, 0.1},
                      {0.2, 0.2, 0.0, 0.6},
                      {0.4, 0.4, 0.2, 0.0}};

    // Instantiating the emission process.
    vec means = {0, 5, 10, 15};
    vec std_devs =  {0.5, 1.0, 0.1, 2.0};
    shared_ptr<AbstractEmission> ptr_emission(new DummyGaussianEmission(
            means, std_devs));

    // Multivariate emission.
    mat mult_means(nstates, 2, fill::zeros);
    for(int i = 0; i < nstates; i++)
        mult_means.row(i).fill(i);
    shared_ptr<AbstractEmission> ptr_mult_emission(
            new DummyMultivariateGaussianEmission(mult_means, 0.1));
    mat mult_sample = ptr_mult_emission->sampleFromState(0, 5);

    // Instantiating the HSMM.
    HSMM dhsmm(ptr_emission, transition, pi, durations, min_duration);

    ivec hiddenStates, hiddenDurations;
    int nSampledSegments = 50;
    mat samples = dhsmm.sampleSegments(nSampledSegments, hiddenStates,
            hiddenDurations);
    int nobs = samples.n_cols;

    cout << "Generated samples" << endl;
    // cout << samples << endl;
    cout << "Generated states and durations" << endl;
    cout << join_horiz(hiddenStates, hiddenDurations) << endl;

    mat alpha(nstates, nobs, fill::zeros);
    mat beta(nstates, nobs, fill::zeros);
    mat alpha_s(nstates, nobs, fill::zeros);
    mat beta_s(nstates, nobs, fill::zeros);
    vec beta_s_0(nstates, fill::zeros);
    cube eta(nstates, ndurations, nobs, fill::zeros);
    cube logpdf = dhsmm.computeEmissionsLogLikelihood(samples);
    logsFB(log(transition), log(pi), log(durations), logpdf, alpha, beta,
            alpha_s, beta_s, beta_s_0, eta, min_duration, nobs);
    mat compare_alpha = exp(alpha);
    mat compare_alpha_s = exp(alpha_s);
    mat compare_beta = exp(beta);
    mat compare_beta_s = exp(beta_s);
    cube compare_eta = exp(eta);
    mat compare_beta_s_0 = exp(beta_s_0);
    cube pdf = dhsmm.computeEmissionsLikelihood(samples);
    FB(transition, pi, durations, pdf, alpha, beta, alpha_s, beta_s, beta_s_0,
            eta, min_duration, nobs);

    cout << "TEST" << endl;
    mat a = compare_beta - beta;
    mat b = compare_beta_s - beta_s;
    cube c = compare_eta - eta;
    mat d = compare_beta_s_0 - beta_s_0;
    mat e = compare_alpha - alpha;
    mat f = compare_alpha_s - alpha_s;
    cout << a.min() << " " << a.max() << endl;
    cout << b.min() << " " << b.max() << endl;
    cout << c.min() << " " << c.max() << endl;
    cout << d.min() << " " << d.max() << endl;
    cout << e.min() << " " << e.max() << endl;
    cout << f.min() << " " << f.max() << endl;

    cout << "Sums rows" << endl;
    cout << sum(alpha, 1) << endl;
    cout << sum(alpha_s, 1) << endl;
    cout << sum(beta, 1) << endl;
    cout << sum(beta_s, 1) << endl;

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
    int differences = 0;
    if (viterbiStates.n_elem == hiddenStates.n_elem) {
        for(int t = 0; t < viterbiStates.n_elem; t++)
            if (hiddenStates(t) != viterbiStates(t))
                differences++;
        cout << " Differences: " << differences << endl;
    }
    else
        cout << "The dimensions don't match." << endl;

    // Initializing uniformly the transitions, initial state pmf and durations.
    transition.fill(1.0/(nstates-1));
    transition.diag().zeros();  // No self-loops.
    dhsmm.setTransition(transition);
    pi.fill(1.0/nstates);
    dhsmm.setPi(pi);
    durations.fill(1.0/ndurations);
    dhsmm.setDuration(durations);

    // Resetting emission parameters.
    vec new_means = {0.1, 0.2, 0.3, 30};
    vec new_std_devs = ones<vec>(nstates) * 10;
    shared_ptr<AbstractEmission> init_emission(new DummyGaussianEmission(
            new_means, new_std_devs));
    dhsmm.setEmission(init_emission);

    cout << "Best transition matrix we can aim at:" << endl;
    mat prueba(nstates, nstates, fill::zeros);
    for(int i = 0; i < hiddenStates.n_elem - 1; i++)
        prueba(hiddenStates(i), hiddenStates(i + 1))++;
    mat pruebasum = sum(prueba, 1);
    for(int i = 0; i < nstates; i++)
        prueba.row(i) /= pruebasum(i);
    cout << prueba << endl;

    // Testing the learning algorithm.
    dhsmm.fit(samples, 100, 1e-10);
    cout << "Learnt matrix:" << endl;
    cout << dhsmm.transition_ << endl;

    cout << "Best duration matrix we can aim at:" << endl;
    mat emp_durations(nstates, ndurations, fill::zeros);
    for(int i = 0; i < hiddenStates.n_elem; i++)
        emp_durations(hiddenStates(i), hiddenDurations(i) - min_duration)++;
    mat emp_durations_sum = sum(emp_durations, 1);
    for(int i = 0; i < nstates; i++)
        emp_durations.row(i) /= emp_durations_sum(i);
    cout << emp_durations << endl;
    cout << "Learnt durations:" << endl;
    cout << dhsmm.duration_ << endl;

    cout << "Learnt pi:" << endl;
    cout << dhsmm.pi_ << endl;

    cout << "Learnt emission parameters" << endl;
    dhsmm.emission_->printParameters();
    return 0;
}