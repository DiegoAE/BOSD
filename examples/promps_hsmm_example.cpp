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
        ivec hiddenStates, ivec hiddenDurations) {
    cout << "Best transition matrix we can aim at:" << endl;
    mat prueba(nstates, nstates, fill::zeros);
    for(int i = 0; i < hiddenStates.n_elem - 1; i++)
        prueba(hiddenStates(i), hiddenStates(i + 1))++;
    mat pruebasum = sum(prueba, 1);
    for(int i = 0; i < nstates; i++)
        prueba.row(i) /= pruebasum(i);
    cout << prueba << endl;

    cout << "Best duration matrix we can aim at:" << endl;
    mat emp_durations(nstates, ndurations, fill::zeros);
    for(int i = 0; i < hiddenStates.n_elem; i++)
        emp_durations(hiddenStates(i), hiddenDurations(i) - min_duration)++;
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
        new_Sigma_w *= 10 * (i + 1);
        mat new_Sigma_y(size(new_model.get_Sigma_y()), fill::eye);
        new_Sigma_y *= 10;
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
    vec pi(nstates, fill::eye);
    pi.fill(1.0/nstates);
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

    int nsegments = 50;
    ivec hidden_states, hidden_durations;
    mat toy_obs = promp_hsmm.sampleSegments(nsegments, hidden_states,
            hidden_durations);
    cout << "Generated states and durations" << endl;
    cout << join_horiz(hidden_states, hidden_durations) << endl;

    PrintBestWeCanAimFor(nstates, ndurations, min_duration, hidden_states,
            hidden_durations);

    // Learning the model from data.
    reset(promp_hsmm, promps);
    promp_hsmm.emission_->to_stream();
    promp_hsmm.fit(toy_obs, 100, 1e-10);
    json params = promp_hsmm.emission_->to_stream();
    cout << params.dump(4) << endl;

    // Running the Viterbi algorithm.
    imat psi_duration(nstates, toy_obs.n_cols, fill::zeros);
    imat psi_state(nstates, toy_obs.n_cols, fill::zeros);
    mat delta(nstates, toy_obs.n_cols, fill::zeros);
    cube pdf = promp_hsmm.computeEmissionsLikelihood(toy_obs);
    Viterbi(transition, pi, durations, pdf, delta, psi_duration, psi_state,
            min_duration, toy_obs.n_cols);
    ivec viterbiStates, viterbiDurations;
    viterbiPath(psi_duration, psi_state, delta, viterbiStates,
            viterbiDurations);

    cout << "Viterbi states and durations" << endl;
    cout << join_horiz(viterbiStates, viterbiDurations) << endl;
    int dur_diff = 0;
    for(int i = 0; i < viterbiDurations.n_elem; i++)
        dur_diff += (viterbiDurations[i] != hidden_durations[i]);
    cout << "The number of mismatches in duration is " << dur_diff << endl;
    return 0;
}
