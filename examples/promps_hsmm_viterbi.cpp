#include <armadillo>
#include <HSMM.hpp>
#include <json.hpp>
#include <ProMPs_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;
using json = nlohmann::json;


int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout<<"Usage: "<< argv[0] <<
                " <input_obs_filename> <input_json_params_filename>\n";
        return 1;
    }
    mat obs;
    obs.load(argv[1], raw_ascii);
    int njoints = obs.n_rows;
    int nobs = obs.n_cols;
    cout << "Time series shape: (" << njoints << ", " << nobs << ")." << endl;
    int min_duration = 45;
    int nstates = 10;
    int ndurations = 10;
    mat transition(nstates, nstates);
    transition.fill(1.0 / (nstates - 1));
    transition.diag().zeros(); // No self-transitions.
    vec pi(nstates);
    pi.fill(1.0/nstates);
    mat durations(nstates, ndurations);
    durations.fill(1.0 / ndurations);
    int n_basis_functions = 4;

    // Setting a third order polynomial basis function for the ProMP
    int polynomial_order = n_basis_functions - 1;
    shared_ptr<ScalarBasisFun> kernel{ new ScalarPolyBasis(polynomial_order)};

    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w(n_basis_functions * njoints);
        mu_w.randn();
        mat Sigma_w = eye<mat>(n_basis_functions * njoints,
                    n_basis_functions * njoints);
        mat Sigma_y = 0.01*eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(kernel, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission and parsing the model parameters as json.
    shared_ptr<AbstractEmission> ptr_emission(new ProMPsEmission(promps));
    HSMM promp_hsmm(ptr_emission, transition, pi, durations, min_duration);
    ifstream input_params_file(argv[2]);
    json input_params;
    input_params_file >> input_params;
    promp_hsmm.from_stream(input_params);

    // Running the Viterbi algorithm.
    imat psi_duration(nstates, nobs, fill::zeros);
    imat psi_state(nstates, nobs, fill::zeros);
    mat delta(nstates, nobs, fill::zeros);
    cout << "Before pdf" << endl;
    cube log_pdf = promp_hsmm.computeEmissionsLogLikelihood(obs);
    cout << "After pdf" << endl;
    Viterbi(promp_hsmm.transition_, promp_hsmm.pi_, promp_hsmm.duration_,
            log_pdf, delta, psi_duration, psi_state, promp_hsmm.min_duration_,
            nobs);
    cout << "Delta last column" << endl;
    cout << delta.col(nobs - 1) << endl;
    ivec viterbiStates, viterbiDurations;
    viterbiPath(psi_duration, psi_state, delta, viterbiStates, viterbiDurations);
    cout << "Viterbi states and durations" << endl;
    cout << join_horiz(viterbiStates, viterbiDurations) << endl;
    return 0;
}
