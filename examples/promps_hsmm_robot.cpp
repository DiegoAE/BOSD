#include <armadillo>
#include <json.hpp>
#include <HSMM.hpp>
#include <ProMPs_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;

// Running the Viterbi algorithm.
void ViterbiAlgorithm(HSMM& promp_hsmm, const mat& obs, const char* filename) {
    int nstates = promp_hsmm.nstates_;
    int nobs = obs.n_cols;
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
    imat states_and_durations = join_horiz(viterbiStates, viterbiDurations);
    cout << states_and_durations << endl;
    cout << "Python states list representation" << endl;
    cout << "[";
    for(int i = 0; i < viterbiStates.n_elem; i++)
        cout << viterbiStates[i] <<
                ((i + 1 == viterbiStates.n_elem)? "]" : ",");
    cout << endl;
    cout << "Python duration list representation" << endl;
    cout << "[";
    for(int i = 0; i < viterbiDurations.n_elem; i++)
        cout << viterbiDurations[i] <<
                ((i + 1 == viterbiDurations.n_elem)? "]" : ",");
    cout << endl;

    // Saving the matrix of joint states and durations.
    states_and_durations.save(filename, raw_ascii);
}

int main(int argc, char *argv[]) {
    if (argc != 4 && argc != 5) {
        cout<<"Usage: "<< argv[0] <<
                " <input_obs_filename> <output_json_params_filename>" <<
                " <output_viterbi> [git_commit_id]\n";
        return 1;
    }
    mat obs;
    obs.load(argv[1], raw_ascii);
    int njoints = obs.n_rows;
    int nobs = obs.n_cols;
    cout << "Time series shape: (" << njoints << ", " << nobs << ")." << endl;
    int min_duration = 30;
    int nstates = 4;
    int ndurations = 50;
    mat transition(nstates, nstates);
    transition.fill(1.0 / nstates );
    vec pi(nstates);
    pi.fill(1.0/nstates);
    mat durations(nstates, ndurations);
    durations.fill(1.0 / ndurations);

    // Setting a combination of polynomial and rbf basis functions.
    auto rbf = shared_ptr<ScalarGaussBasis>(new ScalarGaussBasis(
                {0.25,0.5,0.75},0.25));
    auto poly = make_shared<ScalarPolyBasis>(1);
    auto comb = shared_ptr<ScalarCombBasis>(new ScalarCombBasis({rbf, poly}));
    int n_basis_functions = comb->dim();

    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w(n_basis_functions * njoints);
        mu_w.randn();
        mat Sigma_w = eye<mat>(n_basis_functions * njoints,
                    n_basis_functions * njoints);
        mat Sigma_y = 0.001 * eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(comb, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission.
    shared_ptr<ProMPsEmission> ptr_emission(new ProMPsEmission(promps));

    // Creating a prior for Sigma_w.
    mat Phi = 0.1 * eye<mat>(n_basis_functions * njoints,
            n_basis_functions * njoints);
    InverseWishart iw_prior(Phi, Phi.n_rows + 2);
    ptr_emission->set_Sigma_w_Prior(iw_prior);

    HSMM promp_hsmm(std::static_pointer_cast<AbstractEmission>(ptr_emission),
            transition, pi, durations, min_duration);

    // Saving the model in a json file.
    std::ofstream initial_params(argv[2]);
    nlohmann::json initial_model = promp_hsmm.to_stream();
    if (argc == 5)
        initial_model["git_commit_id"] = argv[4];
    initial_params << std::setw(4) << initial_model << std::endl;
    initial_params.close();
    
    for(int i = 0; i < 10; i++) {

        // Reading the current parameters.
        std::ifstream current_params_stream(argv[2]);
        nlohmann::json current_params;
        current_params_stream >> current_params;
        promp_hsmm.from_stream(current_params);
        bool convergence_reached = promp_hsmm.fit(obs, 5, 1e-5);

        // Saving again the parameters after one training iteration.
        std::ofstream output_params(argv[2]);
        output_params << std::setw(4) << promp_hsmm.to_stream() << std::endl;
        output_params.close();

        ViterbiAlgorithm(promp_hsmm, obs, argv[3]);

        if (convergence_reached)
            break;

    }
    return 0;
}

