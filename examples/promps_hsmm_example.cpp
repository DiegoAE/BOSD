#include <armadillo>
#include <ForwardBackward.hpp>
#include <HSMM.hpp>
#include <iostream>
#include <json.hpp>
#include <memory>
#include <ProMPs_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace std;
using json = nlohmann::json;


void PrintBestWeCanAimFor(int nstates, int ndurations, int min_duration,
        field<ivec> hiddenStates, field<ivec> hiddenDurations) {
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

    cout << "Best duration matrix we can aim at:" << endl;
    mat emp_durations(nstates, ndurations, fill::zeros);
    for(int s = 0; s < nseq; s++) {
        for(int i = 0; i < hiddenStates(s).n_elem; i++)
            emp_durations(hiddenStates(s)(i), hiddenDurations(s)(i)
                    - min_duration)++;
    }
    mat emp_durations_sum = sum(emp_durations, 1);
    for(int i = 0; i < nstates; i++)
        emp_durations.row(i) /= emp_durations_sum(i);
    cout << emp_durations << endl;
}

void reset(HSMM& hsmm, vector<FullProMP> promps) {
    int nstates = hsmm.nstates_;
    int ndurations = hsmm.ndurations_;
    mat transition(hsmm.transition_);
    if (nstates == 1)
        transition.fill(1.0);  // Self-loops allowed in this case.
    else {
        transition.fill(1.0/(nstates-1));
        transition.diag().zeros();  // No self-loops.
    }
    hsmm.setTransition(transition);
    vec pi(hsmm.pi_);
    pi.fill(1.0/nstates);
    hsmm.setPi(pi);
    mat durations(hsmm.duration_);
    durations.fill(1.0/ndurations);
    hsmm.setDuration(durations);

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
    hsmm.setEmission(ptr_emission);
}

int main() {
    int min_duration = 50;
    mat transition = {{0.0, 0.1, 0.4, 0.5},
                      {0.3, 0.0, 0.6, 0.1},
                      {0.2, 0.2, 0.0, 0.6},
                      {0.4, 0.4, 0.2, 0.0}};
    int nstates = transition.n_rows;
    vec pi = {0.1, 0.2, 0.3, 0.4};
    // mat durations(nstates, ndurations, fill::eye);
    mat durations =  {{0.0, 0.1, 0.4, 0.5},
                      {0.3, 0.0, 0.6, 0.1},
                      {0.2, 0.2, 0.0, 0.6},
                      {0.4, 0.4, 0.2, 0.0}};
    int ndurations = durations.n_cols;
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
    shared_ptr<AbstractEmission> ptr_emission(new ProMPsEmission(promps));

    HSMM promp_hsmm(ptr_emission, transition, pi, durations, min_duration);

    int nseq = 1;
    int nsegments = 50;
    field<ivec> hidden_states, hidden_durations;
    field<mat> multiple_toy_obs = promp_hsmm.sampleMultipleSequences(nseq, nsegments,
            hidden_states, hidden_durations);
    cout << "Generated states and durations for the first sequence" << endl;
    cout << join_horiz(hidden_states(0), hidden_durations(0)) << endl;

    PrintBestWeCanAimFor(nstates, ndurations, min_duration, hidden_states,
            hidden_durations);

    cout << "Original emission parameters" << endl;
    json params_test = promp_hsmm.emission_->to_stream();
    cout << params_test.dump(4) << endl;

    reset(promp_hsmm, promps);

    cout << "Emission parameters after reset" << endl;
    params_test = promp_hsmm.emission_->to_stream();
    cout << params_test.dump(4) << endl;

    // Providing some sparse labels.
    set<int> observed_indexes = {};  // {8,25,32,39,47};
    field<Labels> mlabels(nseq);
    for(int s = 0; s < nseq; s++) {
        int idx = 0;
        Labels observed_segments;
        for(int i = 0; i < nsegments; i++) {
            int hs = hidden_states(0)(i);
            int dur = hidden_durations(0)(i);
            idx += dur;
            if (observed_indexes.find(i) != observed_indexes.end()) {
                cout << "label hs: " << hs << endl;
                observed_segments.setLabel(idx - 1, dur, hs);
            }
        }
        mlabels(s) = observed_segments;
    }

    // Learning the model from data.
    promp_hsmm.fit(multiple_toy_obs, mlabels, 100, 1e-10);

    cout << "Emission parameters after training" << endl;
    json params = promp_hsmm.emission_->to_stream();
    cout << params.dump(4) << endl;

    // Running the Viterbi algorithm for the first sequence.
    const mat& toy_obs = multiple_toy_obs(0);
    imat psi_duration(nstates, toy_obs.n_cols, fill::zeros);
    imat psi_state(nstates, toy_obs.n_cols, fill::zeros);
    mat delta(nstates, toy_obs.n_cols, fill::zeros);
    cube log_pdf = promp_hsmm.computeEmissionsLogLikelihood(toy_obs);
    Viterbi(transition, pi, durations, log_pdf, delta, psi_duration, psi_state,
            min_duration, toy_obs.n_cols);
    cout << "Delta last column" << endl;
    cout << delta.col(toy_obs.n_cols - 1) << endl;
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
