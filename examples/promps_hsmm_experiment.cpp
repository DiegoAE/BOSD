#include <armadillo>
#include <HSMM.hpp>
#include <ProMPs_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout<<"Usage: "<< argv[0] <<
                " <input_obs_filename> <output_json_params_filename>\n";
        return 1;
    }
    mat obs;
    obs.load(argv[1], raw_ascii);
    int njoints = obs.n_rows;
    int nobs = obs.n_cols;
    cout << "Time series shape: (" << njoints << ", " << nobs << ")." << endl;
    int min_duration = 45;
    int nstates = 7;
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
    promp_hsmm.fit(obs, 100, 1e-10);

    // Saving the model in a json file.
    std::ofstream output_params(argv[2]);
    output_params << std::setw(4) << promp_hsmm.to_stream() << std::endl;
    return 0;
}