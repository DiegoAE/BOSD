#include <armadillo>
#include <HMM.hpp>
#include <iostream>
#include <json.hpp>
#include <memory>
#include <ProMPs_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace std;
using json = nlohmann::json;


void PrintBestWeCanAimFor(int nstates, field<ivec> hiddenStates) {
    int nseq = hiddenStates.n_elem;
    vec best_pi(nstates, fill::zeros);
    for(int s = 0; s < nseq; s++)
        best_pi(hiddenStates(s)(0))++;
    best_pi /= nseq;
    cout << "Best initial state pmf we can aim at:" << endl << best_pi << endl;

    cout << "Best transition matrix we can aim at:" << endl;
    mat prueba(nstates, nstates, fill::zeros);
    for(int s = 0; s < nseq; s++) {
        for(int i = 0; i < hiddenStates(s).n_elem - 1; i++)
            prueba(hiddenStates(s)(i), hiddenStates(s)(i + 1))++;
    }
    mat pruebasum = sum(prueba, 1);
    for(int i = 0; i < nstates; i++)
        prueba.row(i) /= pruebasum(i);
    cout << prueba << endl;
}

void reset(HMM& hmm, vector<FullProMP> promps) {
    int nstates = hmm.nstates_;
    mat transition(hmm.transition_);
    if (nstates == 1)
        transition.fill(1.0);  // Self-loops allowed in this case.
    else {
        transition.fill(1.0/(nstates-1));
        transition.diag().zeros();  // No self-loops.
    }
    hmm.setTransition(transition);
    vec pi(hmm.pi_);
    pi.fill(1.0/nstates);
    hmm.setPi(pi);

    // Resetting emission.
    for(int i = 0; i < nstates; i++) {
        ProMP new_model = promps[i].get_model();
        vec new_mean = randn(size(new_model.get_mu_w()));
        mat new_Sigma_w(size(new_model.get_Sigma_w()), fill::eye);
        new_Sigma_w *= 10000;
        mat new_Sigma_y(size(new_model.get_Sigma_y()), fill::eye);
        new_Sigma_y *= 1;
        new_model.set_mu_w(new_mean);
        new_model.set_Sigma_w(new_Sigma_w);
        new_model.set_Sigma_y(new_Sigma_y);
        promps[i].set_model(new_model);
    }
    shared_ptr<AbstractEmission> ptr_emission(new ProMPsEmission(promps));

    // TODO: estimate the emission parameters.
    //hmm.setEmission(ptr_emission);
}

int main() {
    mat transition = {{0.0, 0.1, 0.4, 0.5},
                      {0.3, 0.0, 0.6, 0.1},
                      {0.2, 0.2, 0.0, 0.6},
                      {0.4, 0.4, 0.2, 0.0}};
    int nstates = transition.n_rows;
    vec pi = {0.1, 0.2, 0.3, 0.4};
    int n_basis_functions = 4;
    int njoints = 1;

    // Setting a third order polynomial basis function for the ProMP
    int polynomial_order = n_basis_functions - 1;
    shared_ptr<ScalarBasisFun> kernel{ new ScalarPolyBasis(polynomial_order)};

    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w(n_basis_functions * njoints);
        mu_w.fill(i * 10);
        mat Sigma_w = (i + 1) * eye<mat>(n_basis_functions * njoints,
                    n_basis_functions * njoints);
        mat Sigma_y = 0.0001*eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(kernel, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission.
    shared_ptr<AbstractEmission> ptr_emission(new ProMPsEmissionHMM(promps));

    HMM promp_hmm(ptr_emission, transition, pi);

    int nseq = 1;
    int nsegments = 50;
    cout << "Number of sequences: " << nseq << endl;
    cout << "Number of segments in each sequence: " << nsegments << endl;
    field<ivec> hidden_states, hidden_durations;
    field<field<mat>> multiple_toy_obs = promp_hmm.sampleMultipleSequences(
            nseq, nsegments, hidden_states, hidden_durations);
    cout << "Duration for each segment (single observation): " <<
            multiple_toy_obs(0)(0).n_cols << endl;
    cout << "Generated states and durations for the first sequence" << endl;
    cout << join_horiz(hidden_states(0), hidden_durations(0)) << endl;
    PrintBestWeCanAimFor(nstates, hidden_states);

    cout << "Original emission parameters" << endl;
    json params_test = promp_hmm.emission_->to_stream();
    cout << params_test.dump(4) << endl;

    reset(promp_hmm, promps);

    cout << "Model parameters after reset" << endl;
    params_test = promp_hmm.to_stream();
    cout << params_test.dump(4) << endl;

    field<Labels> mlabels(nseq);

    // Learning the model from data.
    promp_hmm.fit(multiple_toy_obs, mlabels, 100, 1e-10);

    cout << "Model parameters after training" << endl;
    json params = promp_hmm.to_stream();
    cout << params.dump(4) << endl;

    // Running the Viterbi algorithm for the first sequence.
    const field<mat>& toy_obs = multiple_toy_obs(0);
    imat psi_duration(nstates, toy_obs.n_elem, fill::zeros);
    imat psi_state(nstates, toy_obs.n_elem, fill::zeros);
    mat delta(nstates, toy_obs.n_elem, fill::zeros);
    cube log_pdf = promp_hmm.computeEmissionsLogLikelihood(toy_obs);
    Viterbi(transition, pi, promp_hmm.duration_, log_pdf, delta, psi_duration,
            psi_state, promp_hmm.min_duration_, toy_obs.n_elem);
    cout << "Delta last column" << endl;
    cout << delta.col(toy_obs.n_elem - 1) << endl;
    ivec viterbiStates, viterbiDurations;
    viterbiPath(psi_duration, psi_state, delta, viterbiStates,
            viterbiDurations);

    cout << "Viterbi states and durations" << endl;
    cout << join_horiz(viterbiStates, viterbiDurations) << endl;
    int dur_diff = 0;
    int states_diff = 0;
    for(int i = 0; i < viterbiDurations.n_elem; i++) {
        dur_diff += (viterbiDurations[i] != hidden_durations(0)(i));
        states_diff += (viterbiStates[i] != hidden_states(0)(i));
    }
    cout << "The number of mismatches in duration is " << dur_diff << endl;
    cout << "The number of mismatches in hidden states is " << states_diff <<
            endl;
    return 0;
}
