#include <mlpack/core.hpp>
#include <NN_emission.hpp>
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

mat reverse(const mat& m) {
    mat ret(size(m));
    for(int i = 0; i < m.n_cols; i++)
        ret.col(m.n_cols - 1 - i) = m.col(i);
    return ret;
}

pair<vec, mat> getMeanAndCovFromSamples(const vector<vec>& samples) {
    vec mean(size(samples.at(0)), fill::zeros);
    mat cov(mean.n_elem, mean.n_elem, fill::zeros);
    for(const auto& s: samples) {
        mean += s;
        cov += s * s.t();
    }
    mean = mean * (1.0/samples.size());
    cov = cov * (1.0/samples.size());
    cov = cov - mean * mean.t();
    return make_pair(mean, cov);
}

ivec get_residual_time_from_vit(const imat& vit) {
    vector<int> ret;
    ivec dur = vit.col(1);
    for(int i = 0; i < vit.n_rows; i++) {
        int d = dur(i);
        for(int j = d - 1; j >= 0; j--)
            ret.push_back(j);
    }
    assert (sum(dur) == ret.size());
    return conv_to<ivec>::from(ret);
}

int main(int argc, char *argv[]) {
    mlpack::Log::Info.ignoreInput = false;
    po::options_description desc("Outputs a fitted HSMM+ProMP model assuming"
            " all the hidden variables are known. Thus, no EM algorithm.");
    desc.add_options()
        ("help,h", "Produce help message")
        ("input,i", po::value<string>(), "Path to the input obs")
        ("output,o", po::value<string>(), "Path to the json output params")
        ("viterbilabels,l", po::value<string>(), "Path to the input viterbi "
            "files (labels). Note that the format is viterbi (not labels)")
        ("nstates,s", po::value<int>(), "Number of hidden states")
        ("mindur", po::value<int>(), "Minimum duration of a segment")
        ("ndur", po::value<int>(), "Number of different durations supported")
        ("hiddenunits,u", po::value<int>()->default_value(10),
                "Number of hidden units per hidden layer")
        ("nlayers", po::value<int>()->default_value(1),
                "Number of hidden layers")
        ("nfiles,n", po::value<int>(), "Number of input files to process")
        ("leaveoneout", po::value<int>()->default_value(-1),
                "Leaves the training file associated to this number out")
        ("debug", "Flag for activating debug mode in HSMM")
        ("nodur", "Flag to deactivate the learning of durations")
        ("notrans", "Flag to deactivate the learning of transitions")
        ("nopi", "Flag to deactivate the learning of initial pmf")
        ("polybasisfun", po::value<int>()->default_value(0), "Order of the "
                "poly basis functions")
        ("noselftransitions", "Flag to deactive self transitions")
        ("delta", po::value<double>()->default_value(-1), "If this is given "
                "a value, the model switches to segment agnostic and the "
                "samples are generated according to the provided delta")
        ("ms", po::value<string>(), "File name where a "
                "matrix containing the state marginals will be stored.")
        ("mr", po::value<string>(), "File name where a "
                "matrix containing the run length marginals will be stored.")
        ("ml", po::value<string>(), "File name where a matrix"
                " containing the residual time marginals will be stored.")
        ("md", po::value<string>(), "File name where a "
                "matrix containing the duration marginals will be stored.")
        ("savebasisfunparams", po::value<string>(), "File where the NN weights"
                " will be saved after training")
        ("test,t", po::value<string>(), "Path to the test observation file")
        ("vittest", po::value<string>(), "Path to the test vit file (truth)")
        ("offsettest", po::value<int>()->default_value(0), "this many number "
                "of observations are skipped for the test llk computation.")
        ("nobstest", po::value<int>()->default_value(-1), "Number of obs to "
                "feed into the model from the test file. Defaults to all obs"
                "(i.e., -1)");
    vector<string> required_fields = {"input", "viterbilabels",
            "nfiles", "nstates", "mindur", "ndur"};
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
    string viterbi_filename = vm["viterbilabels"].as<string>();
    int nseq = vm["nfiles"].as<int>();
    int leaveoneout = vm["leaveoneout"].as<int>();
    int nstates = vm["nstates"].as<int>();
    int hidden_units = vm["hiddenunits"].as<int>();
    int nlayers = vm["nlayers"].as<int>();
    double delta = vm["delta"].as<double>();
    vector<mat> obs_for_each_state[nstates];
    vector<mat> times_for_each_state[nstates];
    field<field<mat>> seq_obs(nseq);
    field<Labels> seq_labels(nseq);
    vector<imat> vit_file_for_each_obs;
    for(int i = 0; i < nseq; i++) {
        if (i == leaveoneout)
            continue;
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
        vit_file_for_each_obs.push_back(vit);
        int idx = 0;
        assert(sum(dur) == obs.n_cols);
        for(int j = 0; j < dur.n_rows; j++) {
            mat segment = obs.cols(idx, idx + dur(j) - 1);
            obs_for_each_state[hs(j)].push_back(segment);

            // NOTE: the training time steps are generated according to delta.
            rowvec times = linspace<rowvec>(0, 1.0, dur(j));
            if (delta > 0)
                times = linspace<rowvec>(0, (dur(j)-1)*delta, dur(j));
            times_for_each_state[hs(j)].push_back(times);
            idx += dur(j);
            seq_labels(i).setLabel(idx - 1, dur(j), hs(j));
        }
    }

    // There should be at least one segment for the hidden state 0.
    int njoints = obs_for_each_state[0].at(0).n_rows;

    // Creating the NN emission.
    shared_ptr<NNEmission> ptr_emission(new NNEmission(nstates, njoints));

    // TODO.
    // if (delta > 0.0)
    //    ptr_emission->setDelta(delta);
    mat noise_vars = ones<mat>(njoints, nstates);
    for(int i = 0; i < nstates; i++) {
        mat inputs = join_mats(times_for_each_state[i]);
        mat outputs = join_mats(obs_for_each_state[i]);
        assert (outputs.n_rows == njoints);
        assert(outputs.n_cols == inputs.n_cols);

        // Training the NN.
        ptr_emission->getNeuralNet(i).Train(inputs, outputs);

        // Var of residuals.
        mat predictions;
        ptr_emission->getNeuralNet(i).Predict(inputs, predictions);
        vec residuals = conv_to<vec>::from(outputs - predictions);
        double noise_var = as_scalar(var(residuals));
        cout << "Var: " << noise_var << endl;

        // Only works for 1D.
        noise_vars.col(i) *= 5 * noise_var;
    }
    ptr_emission->setNoiseVar(noise_vars);
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

    // Computing the required statistics from the provided labels if required.
    // NOTE: pmfs with zero mass are allowed in the transtion and duration
    // matrices.
    if (!vm.count("nopi")) {
        pi.fill(0.0);
        for(const auto& vit: vit_file_for_each_obs)
            pi(vit(0, 0)) += 1.0;
        pi = pi * (1.0/accu(pi));
    }
    cout << "Initial pmf" << endl << pi << endl;
    if (!vm.count("notrans")) {
        transition.fill(0.0);
        for(const auto& vit: vit_file_for_each_obs)
            for(int i = 0; i < vit.col(0).n_elem - 1; i++)
                transition(vit(i, 0), vit(i + 1, 0)) += 1.0;
        for(int i = 0; i < nstates; i++)
            if (accu(transition.row(i)) > 1e-7)
                transition.row(i) *= (1.0/accu(transition.row(i)));
    }
    cout << "Transition matrix" << endl << transition << endl;
    if (!vm.count("nodur")) {
        durations.fill(1.0);
        for(const auto& vit: vit_file_for_each_obs)
            for(int i = 0; i < vit.n_rows; i++) {
                durations(vit(i,0), vit(i,1) - min_duration) += 1.0;
                //durations(1 - vit(i,0), vit(i,1) - min_duration) += 1.0;
            }
        for(int i = 0; i < nstates; i++)
            if (accu(durations.row(i)) > 1e-7)
                durations.row(i) *= (1.0/accu(durations.row(i)));
    }
    cout << "Duration matrix" << endl << durations << endl;


    OnlineHSMM promp_hsmm(std::static_pointer_cast<
            AbstractEmissionOnlineSetting>(ptr_emission),
            transition, pi, durations, min_duration);

    if (!vm.count("test")) {
        cout << "No test file was provided" << endl;
        return 0;
    }
    mat obs_for_cond;
    obs_for_cond.load(vm["test"].as<string>(), raw_ascii);
    field<field<mat>> field_obs = {fromMatToField(obs_for_cond)};

    // Saving the model in a json file.
    if (vm.count("output")) {
        string output_filename = vm["output"].as<string>();
        std::ofstream initial_params(output_filename);
        nlohmann::json initial_model = promp_hsmm.to_stream();
        initial_params << std::setw(4) << initial_model << std::endl;
        initial_params.close();
    }

    // Number of test observations to feed in.
    int n_obs_test = vm["nobstest"].as<int>();
    n_obs_test = (n_obs_test < 0) ?  obs_for_cond.n_cols: n_obs_test;
    int offset_test = vm["offsettest"].as<int>();

    // Load ground truth values of residual time is available.
    double residual_times_ll = 0.0;
    int residual_times_ll_neval = 0;
    ivec ground_truth_residual_t;
    if (vm.count("vittest")) {
        imat test_vit;
        test_vit.load(vm["vittest"].as<string>(), raw_ascii);
        ground_truth_residual_t = get_residual_time_from_vit(test_vit);
    }

    // Testing the online inference algorithm.
    mat state_marginals_over_time(nstates, n_obs_test);
    mat runlength_marginals_over_time(min_duration + ndurations,
            n_obs_test);
    mat residualtime_marginals_over_time(min_duration + ndurations,
            n_obs_test);
    mat duration_marginals_over_time(ndurations, n_obs_test);
    vec onlineloglikelihoods(n_obs_test);
    for(int c = 0; c < n_obs_test; c++) {
        cout << "Processing obs idx: " << c << endl;
        promp_hsmm.addNewObservation(obs_for_cond.col(c));
        onlineloglikelihoods(c) = promp_hsmm.getLastOneStepAheadLoglikelihood();
        if (vm.count("ms")) {
            vec s_marginal = promp_hsmm.getStateMarginal();
            state_marginals_over_time.col(c) = s_marginal;
        }
        if (vm.count("mr")) {
            vec r_marginal = promp_hsmm.getRunlengthMarginal();
            runlength_marginals_over_time.col(c) = r_marginal;
        }
        if (vm.count("ml")) {
            vec l_marginal = promp_hsmm.getResidualTimeMarginal();
            residualtime_marginals_over_time.col(c) = l_marginal;
        }
        if (vm.count("md")) {
            vec d_marginal = promp_hsmm.getDurationMarginal();
            duration_marginals_over_time.col(c) = d_marginal;
        }

        // Computing the loglikelihoods of the residual times if needed.
        if (c >= offset_test && !ground_truth_residual_t.empty()) {
            vec r_marginal = promp_hsmm.getResidualTimeMarginal();
            double cll = r_marginal.at(ground_truth_residual_t.at(c));
            residual_times_ll += log(cll);
            residual_times_ll_neval++;
        }
    }

    // Saving the marginals if required.
    if (vm.count("ms"))
        state_marginals_over_time.save(vm["ms"].as<string>(), raw_ascii);
    if (vm.count("mr"))
        runlength_marginals_over_time.save(vm["mr"].as<string>(), raw_ascii);
    if (vm.count("ml"))
        residualtime_marginals_over_time.save(vm["ml"].as<string>(), raw_ascii);
    if (vm.count("md"))
        duration_marginals_over_time.save(vm["md"].as<string>(), raw_ascii);

    // Evaluation the likelihood of the test observation sequence.
    // The batch one uses the HSMM algos whereas the online takes into account
    // that the last segment might not be complete.
    // double batch_ll = promp_hsmm.loglikelihood(field_obs);
    // cout << "batch loglikelihood: " << batch_ll << endl;
    double online_ll = accu(onlineloglikelihoods);
    cout << "online loglikelihood: " << online_ll << endl;

    // Outputting the llk of the residual times if needed.
    if (!ground_truth_residual_t.empty()) {
        cout << "evaluated terms: " << residual_times_ll_neval << endl;
        cout << "residual times loglikelihood: " << residual_times_ll << endl;
    }
    return 0;
}

