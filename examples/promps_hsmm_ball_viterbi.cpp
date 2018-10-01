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

field<mat> fromMatToField(const mat& obs) {
    field<mat> ret(obs.n_cols);
    for(int i = 0; i < obs.n_cols; i++)
        ret(i) = obs.col(i);
    return ret;
}

// Running the Viterbi algorithm.
void ViterbiAlgorithm(HSMM& promp_hsmm, const field<field<mat>>& seq_obs,
        string filename) {
    int nseq = seq_obs.n_elem;
    for(int s = 0; s < nseq; s++) {
        const field<mat>& obs = seq_obs(s);
        int nstates = promp_hsmm.nstates_;
        int nobs = obs.n_elem;
        imat psi_duration(nstates, nobs, fill::zeros);
        imat psi_state(nstates, nobs, fill::zeros);
        mat delta(nstates, nobs, fill::zeros);
        cout << "Before pdf" << endl;
        cube log_pdf = promp_hsmm.computeEmissionsLogLikelihood(obs);
        cout << "After pdf" << endl;
        Viterbi(promp_hsmm.transition_, promp_hsmm.pi_, promp_hsmm.duration_,
                log_pdf, delta, psi_duration, psi_state,
                promp_hsmm.min_duration_, nobs);
        cout << "Delta last column" << endl;
        cout << delta.col(nobs - 1) << endl;
        ivec viterbiStates, viterbiDurations;
        viterbiPath(psi_duration, psi_state, delta, viterbiStates,
                viterbiDurations);
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
        string viterbi_filename(filename);
        if (nseq > 1)
            viterbi_filename += string(".") + to_string(s);
        states_and_durations.save(viterbi_filename, raw_ascii);
    }
}

int main(int argc, char *argv[]) {
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("input,i", po::value<string>(), "Path to the input obs")
        ("params,p", po::value<string>(), "Path to the json input params")
        ("output,o", po::value<string>(), "Path to the output viterbi file(s)")
        ("polybasisfun", po::value<int>()->default_value(1), "Order of the "
                "poly basis functions")
	    ("norbf", "Flag to deactivate the radial basis functions")
        ("nsequences,n", po::value<int>()->default_value(1),
                "Number of sequences used for training");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }
    if (!vm.count("input") || !vm.count("output") || !vm.count("params")) {
        cerr << "Error: You should provide input and output files" << endl;
        return 1;
    }
    string input_filename = vm["input"].as<string>();
    string output_filename = vm["output"].as<string>();
    string params = vm["params"].as<string>();
    int nseq = vm["nsequences"].as<int>();

    field<field<mat>> seq_obs(nseq);
    int njoints;
    for(int i = 0; i < nseq; i++) {
        string name = input_filename;
        if (nseq != 1)
            name += string(".") + to_string(i);
        mat obs;
        obs.load(name, raw_ascii);
        ifstream input_params_file(params);
        json input_params;
        input_params_file >> input_params;
        njoints = obs.n_rows;
        int nobs = obs.n_cols;
        cout << "Time series shape: (" << njoints << ", " << nobs << ")." << endl;
        seq_obs(i) = fromMatToField(obs);
    }

    ifstream input_params_file(params);
    json input_params;
    input_params_file >> input_params;
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

    // Setting a combination of polynomial and rbf basis functions.
    auto rbf = shared_ptr<ScalarGaussBasis>(new ScalarGaussBasis(
                {0.25,0.5,0.75},0.25));
    auto poly = make_shared<ScalarPolyBasis>(vm["polybasisfun"].as<int>());
    auto comb = shared_ptr<ScalarCombBasis>(new ScalarCombBasis({rbf, poly}));
    if (vm.count("norbf"))
        comb = shared_ptr<ScalarCombBasis>(new ScalarCombBasis({poly}));
    int n_basis_functions = comb->dim();

    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w(n_basis_functions * njoints);
        mu_w.randn();
        mat Sigma_w = eye<mat>(n_basis_functions * njoints,
                    n_basis_functions * njoints);
        mat Sigma_y = 0.01*eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(comb, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission and parsing the model parameters as json.
    shared_ptr<AbstractEmission> ptr_emission(new ProMPsEmission(promps));
    HSMM promp_hsmm(ptr_emission, transition, pi, durations, min_duration);
    promp_hsmm.from_stream(input_params);

    ViterbiAlgorithm(promp_hsmm, seq_obs, output_filename);
    return 0;
}
