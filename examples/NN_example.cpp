#include <mlpack/core.hpp>
#include <NN_basis_function.hpp>
#include <armadillo>
#include <boost/program_options.hpp>
#include <iostream>
#include <robotics.hpp>
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

mat join_mats(vector<mat> v) {
    int total_cols = 0;
    int total_rows = v.at(0).n_rows;
    for(auto& m: v)
        total_cols += m.n_cols;
    mat ret(total_rows, total_cols);
    int idx = 0;
    for(auto& m: v) {
        int d = m.n_cols;
        ret.cols(idx, idx + d - 1) = m;
        idx += d;
    }
    assert(idx == total_cols);
    return ret;
}

int main(int argc, char *argv[]) {
    mlpack::Log::Info.ignoreInput = false;  // Turning mlpack verbose output on.
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("input,i", po::value<string>(), "Path to the input obs")
        ("output,o", po::value<string>(), "Path to the json output params")
        ("viterbi,v", po::value<string>(), "Path to the input viterbi file")
        ("nstates,s", po::value<int>(), "Number of states (NNs)")
        ("mindur", po::value<int>(), "Minimum duration of a segment")
        ("ndur", po::value<int>(), "Number of different durations supported")
        ("hiddenunits,u", po::value<int>()->default_value(10),
                "Number of hidden units")
        ("nfiles,n", po::value<int>()->default_value(1),
                "Number of input files to process")
        ("debug", "Flag for activating debug mode in HSMM")
        ("nodur", "Flag to deactivate the learning of durations")
        ("notrans", "Flag to deactivate the learning of transitions")
        ("nopi", "Flag to deactivate the learning of initial pmf")
        ("durmomentmatching", "Flag to active the Gaussian moment matching"
                " for the duration learning")
        ("noselftransitions", "Flag to deactive self transitions")
        ("wpriorvar", po::value<double>(), "Prior variance for Sigma_w")
        ("alphadurprior", po::value<int>()->default_value(1),
                "Alpha for Dirichlet prior for the duration")
        ("trainingiter", po::value<int>()->default_value(10), "Training "
                "iterations")
        ("delta", po::value<double>(), "If this is given a value, the model "
                "switches to segment agnostic and the samples are generated "
                "according to the provided delta")
        ("leaveoneout", po::value<int>(), "Index of the sequence that will be"
                " left out for validation");
    vector<string> required_fields = {"input", "output", "viterbi", "nfiles",
            "nstates", "mindur", "ndur"};
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << endl;
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
    string viterbi_filename = vm["viterbi"].as<string>();
    int nseq = vm["nfiles"].as<int>();
    int nstates = vm["nstates"].as<int>();
    int hidden_units = vm["hiddenunits"].as<int>();
    vector<mat> obs_for_each_state[nstates];
    vector<mat> times_for_each_state[nstates];
    field<field<mat>> seq_obs(nseq);
    field<Labels> seq_labels(nseq);
    for(int i = 0; i < nseq; i++) {
        string iname = input_filename;
        string vname = viterbi_filename;
        if (nseq != 1) {
            iname += string(".") + to_string(i);
            vname += string(".") + to_string(i);
        }
        mat obs;
        obs.load(iname, raw_ascii);
        seq_obs(i) = fromMatToField(obs);
        imat vit;
        vit.load(vname, raw_ascii);
        ivec hs = vit.col(0);
        ivec dur = vit.col(1);
        int idx = 0;
        for(int j = 0; j < dur.n_rows; j++) {
            mat segment = obs.cols(idx, idx + dur(j) - 1);
            obs_for_each_state[hs(j)].push_back(segment);
            rowvec times = linspace<rowvec>(0, 1.0, dur(j));
            times_for_each_state[hs(j)].push_back(times);
            idx += dur(j);
            seq_labels(i).setLabel(idx - 1, dur(j), hs(j));
        }
    }

    // There should be at least one segment for the hidden state 0.
    int njoints = obs_for_each_state[0].at(0).n_rows;
    vector<shared_ptr<ScalarNNBasis>> nns;
    for(int i = 0; i < nstates; i++) {
        mat inputs = join_mats(times_for_each_state[i]);
        mat outputs = join_mats(obs_for_each_state[i]);
        assert (outputs.n_rows == njoints);
        assert(outputs.n_cols == inputs.n_cols);

        // Defining the architecture of the NN.
        auto nn = make_shared<ScalarNNBasis>(hidden_units, njoints);

        // Training the NN.
        nn->getNeuralNet().Train(inputs, outputs);

        // Predicting.
        vec test_input = linspace<vec>(0,1,100);
        vector<mat> test_output;
        for(int j = 0; j < test_input.n_elem; j++) {
            vec input = nn->eval(test_input(j));
            test_output.push_back(input);
        }
        mat mat_test_output = join_mats(test_output);
        //mat_test_output.save("prediction.txt." + to_string(i) , raw_ascii);
        nns.push_back(nn);
    }
    int n_basis_functions = nns.at(0)->dim();

    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w(n_basis_functions * njoints);
        mu_w.randn();
        mat Sigma_w = eye<mat>(n_basis_functions * njoints,
                    n_basis_functions * njoints);
        mat Sigma_y = 0.001 * eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP nn_promp(nns.at(i), promp, njoints);
        promps.push_back(nn_promp);
    }
    int min_duration = vm["mindur"].as<int>();
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

    HSMM promp_hsmm(std::static_pointer_cast<AbstractEmission>(ptr_emission),
            transition, pi, durations, min_duration);

    if (vm.count("nodur"))
        promp_hsmm.setDurationLearningChoice("nodur");
    if (vm.count("durmomentmatching"))
        promp_hsmm.setDurationLearningChoice("momentmatching");
    if (vm.count("alphadurprior")) {
        mat alphas = ones<mat>(nstates, ndurations) *
            vm["alphadurprior"].as<int>();
        promp_hsmm.setDurationDirichletPrior(alphas);
    }
    if (vm.count("notrans"))
        promp_hsmm.learning_transitions_ = false;
    if (vm.count("nopi"))
        promp_hsmm.learning_pi_ = false;
    if (vm.count("debug"))
        promp_hsmm.debug_ = true;

    // Saving the model in a json file.
    std::ofstream initial_params(output_filename);
    nlohmann::json initial_model = promp_hsmm.to_stream();
    initial_params << std::setw(4) << initial_model << std::endl;
    initial_params.close();

    // Leave one out.
    field<field<mat>> t_seq;
    field<Labels> t_labels;
    if (vm.count("leaveoneout")) {
        int omitted = vm["leaveoneout"].as<int>();
        field<field<mat>> left_one_out(seq_obs.n_elem - 1);
        field<Labels> left_labels(seq_labels.n_elem - 1);
        int idx = 0;
        for(int i = 0; i < seq_obs.n_elem; i++)
            if (i != omitted) {
                left_one_out(idx) = seq_obs(i);
                left_labels(idx) = seq_labels(i);
                idx++;
            }
        t_seq = left_one_out;
        t_labels = left_labels;
        cout << "Leaving one out of the training: " << omitted << endl;
    }
    else {
        t_seq = seq_obs;
        t_labels = seq_labels;
    }

    for(int i = 0; i < vm["trainingiter"].as<int>(); i++) {

        // Reading the current parameters.
        std::ifstream current_params_stream(output_filename);
        nlohmann::json current_params;
        current_params_stream >> current_params;
        promp_hsmm.from_stream(current_params);

        bool convergence_reached = promp_hsmm.fit(t_seq, t_labels, 5, 1e-5);

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

    cout << "loglikelihood: " << promp_hsmm.loglikelihood(t_seq) << endl;
    if (vm.count("leaveoneout")) {
        int omitted = vm["leaveoneout"].as<int>();
        field<field<mat>> test = {seq_obs(omitted)};
        field<Labels> test_labels = {seq_labels(omitted)};
        cout << "loglikelihoodtest: " << promp_hsmm.loglikelihood(test) << endl;
    }
    return 0;
}
