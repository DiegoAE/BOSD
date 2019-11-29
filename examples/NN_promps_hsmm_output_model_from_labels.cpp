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
        ("test,t", po::value<string>(), "Path to the test observation file");
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
    vector<shared_ptr<ScalarCombBasis>> basis;
    vector<vec> means;
    vector<nlohmann::json> basis_fun_params;
    for(int i = 0; i < nstates; i++) {
        mat inputs = join_mats(times_for_each_state[i]);
        mat outputs = join_mats(obs_for_each_state[i]);
        assert (outputs.n_rows == njoints);
        assert(outputs.n_cols == inputs.n_cols);

        // Defining the architecture of the NN.
        ivec hidden_units_per_layer = ones<ivec>(nlayers) * hidden_units;
        auto nn = make_shared<ScalarNNBasis>(hidden_units_per_layer, njoints);

        // Training the NN.
        nn->getNeuralNet().Train(inputs, outputs);
        auto serialized = nn->to_stream();
        basis_fun_params.push_back(serialized);

        // Testing the NN building from serialized parameters.
        auto nn_copy = make_shared<ScalarNNBasis>(serialized);

        // Extracting the parameters from the output layer.
        pair<mat,vec> out_params = nn->getOutputLayerParams();
        mat joint_params = join_horiz(out_params.first, out_params.second);
        vec flattened_params = conv_to<vec>::from(vectorise(joint_params, 1));
        means.push_back(flattened_params);
        cout << "Weights " << endl << out_params.first << endl;
        cout << "Bias " << endl << out_params.second << endl;

        // Adding polynomial terms to the NN basis.
        auto poly = make_shared<ScalarPolyBasis>(vm["polybasisfun"].as<int>());
        auto comb = shared_ptr<ScalarCombBasis>(new ScalarCombBasis(
                    {nn, poly}));
        // Predicting.
        vec test_input = linspace<vec>(0,1,100);
        vector<mat> test_output;
        for(int j = 0; j < test_input.n_elem; j++) {
            vec output = nn->eval(test_input(j));
            vec output2 = nn_copy->eval(test_input(j));
            assert(approx_equal(output, output2, "reldiff", 1e-8));
            test_output.push_back(output);
        }
        mat mat_test_output;
        mat mat_test_input = linspace<rowvec>(0, 1, 100);
        nn->getNeuralNet().Predict(mat_test_input, mat_test_output);
        //mat_test_output.save("prediction.txt." + to_string(i) , raw_ascii);
        mat prueba = out_params.first * join_mats(test_output);
        for(int i = 0; i < prueba.n_cols; i++)
            prueba.col(i) += out_params.second;
        //prueba.save("fprediction.txt." + to_string(i) , raw_ascii);
        assert(approx_equal(mat_test_output, prueba, "reldiff", 1e-8));
        basis.push_back(comb);
    }
    int n_basis_functions = basis.at(0)->dim();

    if (vm.count("savebasisfunparams")) {
        nlohmann::json basisfunparams = basis_fun_params;
        std::ofstream basis_fun_file(vm["savebasisfunparams"].as<string>());
        basis_fun_file << std::setw(4) << basisfunparams << std::endl;
        basis_fun_file.close();
    }

    // Computing the the mean and covariance of Omega using least squares.
    // NOTE: this is only tested for one dimensional observations.
    vector<pair<vec, mat>> mle_gaussians;
    for(int i = 0; i < nstates; i++) {
        const auto& nn = basis.at(i);
        const vector<mat>& t_seq = times_for_each_state[i];
        const vector<mat>& obs_seq = obs_for_each_state[i];
        vector<vec> ls_omegas;
        for(int j = 0; j < t_seq.size(); j++) {
            vector<mat> Phi_vec, Y_vec;
            for(const auto& time_step: t_seq.at(j))
                Phi_vec.push_back(nn->eval(time_step));
            mat Phi = join_mats(Phi_vec).t();
            mat Y = vectorise(obs_seq.at(j));
            vec ls_omega = solve(Phi, Y);
            ls_omegas.push_back(ls_omega);
            assert(ls_omega.n_rows == n_basis_functions * njoints);
        }
        pair<vec, mat> curr = getMeanAndCovFromSamples(ls_omegas);
        mle_gaussians.push_back(curr);
        //cout << "Mean: " << curr.first << endl;
        //cout << "Covariance: " << curr.second << endl;

        vec test_input = linspace<vec>(0,1,100);
        mat mean(njoints, test_input.n_elem);
        mat std_dev(njoints, test_input.n_elem);
        for(int j = 0; j < test_input.n_elem; j++) {
            vec m = nn->eval(test_input(j));
            vec output_mean = m.t() * curr.first;
            mat output_cov = m.t() * curr.second * m;
            mean.col(j) = output_mean;
            std_dev.col(j) = sqrt(output_cov.diag());
        }

        //Debug
        //mean.save("mean_prediction.txt." + to_string(i), raw_ascii);
        //std_dev.save("stddev_prediction.txt." + to_string(i), raw_ascii);
    }


    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w = mle_gaussians.at(i).first;
        assert(mu_w.n_elem == n_basis_functions * njoints);

        // TODO: the covariance is fixed for now due to overfitting.
        mat Sigma_w = 0.05*eye(mu_w.n_elem, mu_w.n_elem);
        mat Sigma_y = 0.001 * eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP nn_promp(basis.at(i), promp, njoints);
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
        durations.fill(0.0);
        for(const auto& vit: vit_file_for_each_obs)
            for(int i = 0; i < vit.n_rows; i++)
                durations(vit(i,0), vit(i,1) - min_duration) += 1.0;
        for(int i = 0; i < nstates; i++)
            if (accu(durations.row(i)) > 1e-7)
                durations.row(i) *= (1.0/accu(durations.row(i)));
    }
    cout << "Duration matrix" << endl << durations << endl;

    // Creating the ProMP emission.
    shared_ptr<ProMPsEmission> ptr_emission(new ProMPsEmission(promps));

    if (delta > 0.0)
        ptr_emission->setDelta(delta);

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

    // Testing the online inference algorithm.
    mat state_marginals_over_time(nstates, obs_for_cond.n_cols);
    mat runlength_marginals_over_time(min_duration + ndurations,
            obs_for_cond.n_cols);
    mat residualtime_marginals_over_time(min_duration + ndurations,
            obs_for_cond.n_cols);
    mat duration_marginals_over_time(ndurations, obs_for_cond.n_cols);
    vec onlineloglikelihoods(obs_for_cond.n_cols);
    for(int c = 0; c < obs_for_cond.n_cols; c++) {
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
    double batch_ll = promp_hsmm.loglikelihood(field_obs);
    double online_ll = accu(onlineloglikelihoods);
    cout << "batch loglikelihood: " << batch_ll << endl;
    cout << "online loglikelihood: " << online_ll << endl;
    return 0;
}

