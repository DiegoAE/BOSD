#include <armadillo>
#include <boost/program_options.hpp>
#include <HSMM.hpp>
#include <json.hpp>
#include <ProMPs_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;
using json = nlohmann::json;
namespace po = boost::program_options;


int main(int argc, char *argv[]) {
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("input,i", po::value<string>(), "Path to the input obs")
        ("params,p", po::value<string>(), "Path to the json input params")
        ("output,o", po::value<string>(), "Path to the output params")
        ("nbasis,nb", po::value<int>(), "Number of basis functions used");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }
    if (!vm.count("input") || !vm.count("output") || !vm.count("params") ||
            !vm.count("nbasis")) {
        cerr << "Error: You should provide input and output files" << endl;
        return 1;
    }
    string input_filename = vm["input"].as<string>();
    string output_filename = vm["output"].as<string>();
    string params = vm["params"].as<string>();
    int n_basis_functions = vm["nbasis"].as<int>();

    mat obs;
    obs.load(input_filename, raw_ascii);
    ifstream input_params_file(params);
    json input_params;
    input_params_file >> input_params;
    int njoints = obs.n_rows;
    int nobs = obs.n_cols;
    cout << "Time series shape: (" << njoints << ", " << nobs << ")." << endl;
    int min_duration = input_params["min_duration"];
    int nstates = input_params["nstates"];
    int ndurations = input_params["ndurations"];
    mat transition(nstates, nstates);
    transition.fill(1.0 / (nstates - 1));
    transition.diag().zeros(); // No self-transitions.
    vec pi(nstates);
    pi.fill(1.0/nstates);
    mat durations(nstates, ndurations);
    durations.fill(1.0 / ndurations);

    // Setting polynomial basis function for the ProMP
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
    states_and_durations.save(output_filename, raw_ascii);
    return 0;
}
