#include <armadillo>
#include <boost/program_options.hpp>
#include <json.hpp>
#include <HSMM.hpp>
#include <ProMPs_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;
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
        ("output,o", po::value<string>(), "Path to the json output params")
        ("nstates", po::value<int>(), "Number of hidden states for the HSMM")
        ("mindur", po::value<int>(), "Minimum duration of a segment")
        ("ndur", po::value<int>(), "Number of different durations supported")
        ("commitid", po::value<string>(), "Git commit id of the experiment")
        ("labels", po::value<string>(), "Path to the provided labels")
        ("viterbi,v", po::value<string>(), "Path to the output viterbi file")
        ("nfiles", po::value<int>()->default_value(1),
                "Number of files (sequences) to process")
        ("debug", "Flag for activating debug mode in HSMM")
        ("nodur", "Flag to deactivate the learning of durations")
        ("durmomentmatching", "Flag to active the Gaussian moment matching"
                " for the duration learning")
        ("polybasisfun", po::value<int>()->default_value(1), "Order of the "
                "poly basis functions")
        ("noselftransitions", "Flag to deactive self transitions")
        ("initfraction", po::value<double>()->default_value(0.1), "Fraction "
                "of the least squares estimates for omega kept for init")
        ("wpriorvar", po::value<double>(), "Prior variance for Sigma_w")
        ("trainingiter", po::value<int>()->default_value(10), "Training "
                "iterations")
        ("norbf", "Flag to deactivate the radial basis functions")
        ("delta", po::value<double>(), "If this is given a value, the model "
                "switches to segment agnostic and the samples are generated "
                "according to the provided delta");
    vector<string> required_fields = {"input", "output", "nstates", "mindur",
            "ndur", "viterbi"};
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }
    if (vm.count("nodur") && vm.count("durmomentmatching")) {
        cout << "Only one choice for duration learning is allowed" << endl;
        return 0;
    }
    for(auto s: required_fields) {
        if (!vm.count(s)) {
            cerr << "Error: You must provide the argument: " << s << endl;
            return 1;
        }
    }
    string input_filename = vm["input"].as<string>();
    string output_filename = vm["output"].as<string>();
    field<field<mat>> seq_obs(1);
    field<Labels> seq_labels(1);
    int njoints;
    int nseq = vm["nfiles"].as<int>();
    seq_obs.set_size(nseq);
    seq_labels.set_size(nseq);
    for(int i = 0; i < seq_obs.n_elem; i++) {
        string name = input_filename;
        if (nseq != 1)
            name += string(".") + to_string(i);
        mat obs;
        obs.load(name, raw_ascii);
        njoints = obs.n_rows;
        int nobs = obs.n_cols;
        cout << "Time series shape: (" << njoints << ", " << nobs <<
            ")." << endl;
        seq_obs(i) = fromMatToField(obs);

        // Reading labels for different obs.
        if (!vm.count("labels"))
            continue;
        string labels_name = vm["labels"].as<string>();
        if (nseq != 1)
            labels_name += string(".") + to_string(i);
        mat labels_mat;
        labels_mat.load(labels_name);
        for(int j = 0; j < labels_mat.n_rows; j++)
            seq_labels(i).setLabel(labels_mat(j, 0), labels_mat(j, 1),
                    labels_mat(j, 2));
    }

    int min_duration = vm["mindur"].as<int>();
    int nstates = vm["nstates"].as<int>();
    int ndurations = vm["ndur"].as<int>();
    mat transition(nstates, nstates);
    transition.fill(1.0 / nstates );
    if (vm.count("noselftransitions")) {
        transition.fill(1.0 / (nstates - 1));
        transition.diag().zeros();
    }
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
        mat Sigma_y = 0.001 * eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(comb, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission.
    shared_ptr<ProMPsEmission> ptr_emission(new ProMPsEmission(promps));

    // Creating a prior for Sigma_w.
    if (vm.count("wpriorvar")) {
        mat Phi = vm["wpriorvar"].as<double>() * eye<mat>(
                n_basis_functions * njoints, n_basis_functions * njoints);
        NormalInverseWishart iw_prior(Phi, Phi.n_rows + 2);
        ptr_emission->set_Sigma_w_Prior(iw_prior);
    }
    if (vm.count("delta"))
        ptr_emission->setDelta(vm["delta"].as<double>());

    // Settings for the initialization algorithm.
    ptr_emission->setParamsForInitialization(vm["initfraction"].as<double>());

    HSMM promp_hsmm(std::static_pointer_cast<AbstractEmission>(ptr_emission),
            transition, pi, durations, min_duration);
    if (vm.count("nodur"))
        promp_hsmm.setDurationLearningChoice("nodur");
    if (vm.count("durmomentmatching"))
        promp_hsmm.setDurationLearningChoice("momentmatching");
    if (vm.count("debug"))
        promp_hsmm.debug_ = true;

    // Initializing the model from data.
    promp_hsmm.init_params_from_data(seq_obs);

    // Saving the model in a json file.
    std::ofstream initial_params(output_filename);
    nlohmann::json initial_model = promp_hsmm.to_stream();
    if (vm.count("commitid"))
        initial_model["git_commit_id"] = vm["commitid"].as<string>();
    initial_params << std::setw(4) << initial_model << std::endl;
    initial_params.close();
    for(int i = 0; i < vm["trainingiter"].as<int>(); i++) {

        // Reading the current parameters.
        std::ifstream current_params_stream(output_filename);
        nlohmann::json current_params;
        current_params_stream >> current_params;
        promp_hsmm.from_stream(current_params);
        bool convergence_reached = promp_hsmm.fit(seq_obs, seq_labels, 5, 1e-5);

        // Saving again the parameters after one training iteration.
        std::ofstream output_params(output_filename);
        current_params = promp_hsmm.to_stream();
        if (vm.count("commitid"))
            current_params["git_commit_id"] = vm["commitid"].as<string>();
        output_params << std::setw(4) << current_params << std::endl;
        output_params.close();

        ViterbiAlgorithm(promp_hsmm, seq_obs, vm["viterbi"].as<string>());

        if (convergence_reached)
            break;

    }
    cout << "loglikelihood: " << promp_hsmm.loglikelihood(seq_obs) << endl;
    return 0;
}

