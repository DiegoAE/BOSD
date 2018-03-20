#include <armadillo>
#include <json.hpp>
#include <HSMM.hpp>
#include <ProMPs_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 4) {
        cout<<"Usage: "<< argv[0] <<
                " <input_obs_filename> <output_json_params_filename>\n" <<
                "[git_commit_id]\n";
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
        mu_w.randu();
        mat Sigma_w = eye<mat>(n_basis_functions * njoints,
                    n_basis_functions * njoints);
        mat Sigma_y = 0.01*eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(kernel, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission.
    shared_ptr<ProMPsEmission> ptr_emission(new ProMPsEmission(promps));

    // Creating a prior for Sigma_w.
    mat Phi = 0.5 * eye<mat>(n_basis_functions * njoints,
            n_basis_functions * njoints);
    InverseWishart iw_prior(Phi, Phi.n_rows + 2);
    ptr_emission->set_Sigma_w_Prior(iw_prior);

    HSMM promp_hsmm(std::static_pointer_cast<AbstractEmission>(ptr_emission),
            transition, pi, durations, min_duration);

    // Saving the model in a json file.
    std::ofstream output_params(argv[2]);
    nlohmann::json initial_model = promp_hsmm.to_stream();
    if (argc == 4)
        initial_model["git_commit_id"] = argv[3];
    output_params << std::setw(4) << initial_model << std::endl;
    output_params.close();

    for(int i = 0; i < 10; i++) {

        // Reading the current parameters.
        std::ifstream current_params_stream(argv[2]);
        nlohmann::json current_params;
        current_params_stream >> current_params;
        promp_hsmm.from_stream(current_params);

        bool convergence_reached = promp_hsmm.fit(obs, 10, 1e-5);

        // Saving again the parameters after one training iteration.
        output_params.open(argv[2]);
        output_params << std::setw(4) << promp_hsmm.to_stream() << std::endl;
        output_params.close();

        if (convergence_reached)
            break;

    }
    return 0;
}
