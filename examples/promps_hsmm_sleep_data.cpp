#include <armadillo>
#include <boost/program_options.hpp>
#include <HSMM.hpp>
#include <Multivariate_Gaussian_emission.hpp>
#include <random>

using namespace arma;
using namespace hsmm;
using namespace robotics::random;
using namespace std;
namespace po = boost::program_options;


// MLE transition in the HMM case. This means self-transitions are allowed.
mat getHmmTransitionFromLabels(const field<ivec>& labels_seq, int nstates) {
    mat hmm_transition(nstates, nstates, fill::zeros);
    for(const ivec &s : labels_seq)
        for(int i = 0; i < s.n_elem - 1; i++)
            hmm_transition(s(i), s(i + 1))++;

    for(int i = 0; i < nstates; i++)
        hmm_transition.row(i) = hmm_transition.row(i) / accu(
                hmm_transition.row(i));
    return hmm_transition;
}

// Equivalent to np.fft.rfftfreq(512, 1.0 / 128.0) in Python.
vec discreteFourierTransformSampleFrequencies() {
    return linspace(0, 64, 257);
}

vec extract_frequency_features(vec input) {
    vec nu = discreteFourierTransformSampleFrequencies();
    uvec delta_band = find(nu > 0.4 && nu < 4.1);
    uvec theta_band = find(nu > 5.9 && nu < 10.1);
    uvec alpha_band = find(nu > 10 && nu < 15.1);
    uvec all_bands = find(nu > 3.9 && nu < 40.1);

    cx_mat t = fft(input);

    // Ensuring the same size as np.fft.rfft(input).
    t = t.head_rows(input.n_elem / 2 + 1);
    mat norm = abs(t);
    norm = norm % norm;
    vec features = {accu(norm.elem(delta_band)), accu(norm.elem(theta_band)),
            accu(norm.elem(alpha_band)), accu(norm.elem(all_bands))};
    return features;
}

vec extract_eeg_features(vec eeg) {
    return extract_frequency_features(eeg).head_rows(3);
}

vec extract_emg_features(vec emg) {
    return extract_frequency_features(emg).tail_rows(1);
}

field<vec> getFeatureVectors(const mat& eeg1, const mat& eeg2, const mat& emg) {
    int nobs = eeg1.n_cols;
    assert(nobs == eeg2.n_cols);
    assert(nobs == emg.n_cols);
    field<vec> features(nobs);
    for(int i = 0; i < nobs; i++) {
        vec f = join_vert(extract_eeg_features(eeg1.col(i)),
                extract_eeg_features(eeg2.col(i)));
        features(i) = join_vert(f, extract_emg_features(emg.col(i)));
    }
    return features;
}

ivec predict_labels_iid(shared_ptr<MultivariateGaussianEmission> e,
        const field<vec>& test_input, const vec& class_prior) {
    assert(class_prior.n_elem == e->getNumberStates());
    ivec ret(test_input.n_elem);
    for(int i = 0; i < test_input.n_elem; i++) {
        vec loglikelihoods(e->getNumberStates());
        for(int j = 0; j < e->getNumberStates(); j++)
            loglikelihoods(j) = e->loglikelihood(j,
                    test_input(i)) + log(class_prior(j));
        ret(i) = (int) loglikelihoods.index_max();
    }
    return ret;
}

ivec predict_labels_from_filtering(const mat& filtering_state_marginals) {
    ivec ret(filtering_state_marginals.n_cols);
    for(int i = 0; i < ret.n_elem; i++)
        ret(i) = (int) filtering_state_marginals.col(i).index_max();
    return ret;
}

vector<NormalDist> get_normal_distributions(int nstates, int ndurations,
        int min_duration, int ndimension) {
    mt19937 gen(0);
    vector<NormalDist> states;
    vector<vec> samples;
    vector<int> labels;
    for(int i = 0; i < nstates; i++) {
        vec mean = ones<vec>(ndimension) * i * 10;
        mat cov = eye(ndimension, ndimension);
        NormalDist a(mean, cov);
        states.push_back(a);

        // Generating toy data.
        int nsamples = 100;
        vector<vec> s = sample_multivariate_normal(gen, a, nsamples);
        vector<int> l = conv_to<vector<int>>::from(ones<ivec>(nsamples) * i);
        samples.insert(samples.end(), s.begin(), s.end());
        labels.insert(labels.end(), l.begin(), l.end());
    }

    // Toy data handling.
    ivec labels_vec = conv_to<ivec>::from(labels);
    field<vec> obs_field(samples.size());
    for(int i = 0; i < samples.size(); i++)
        obs_field(i) = samples.at(i);

    // Debug
    shared_ptr<MultivariateGaussianEmission> emission(
            new MultivariateGaussianEmission(states));
    OnlineHSMMRunlengthBased model(emission, nstates, ndurations, min_duration);
    for(int i = 0; i < samples.size(); i++)
        model.addNewObservation(samples.at(i));
    return states;
}

int main(int argc, char *argv[]) {
    po::options_description desc("Options");
    vector<string> input_features, input_labels;
    desc.add_options()
        ("help,h", "Produce help message")
        ("input,i", po::value<vector<string>>(&input_features)->multitoken(),
                "Path to the (multiple) input features")
        ("output,o", po::value<string>(), "Path to the json output params")
        ("nstates", po::value<int>(), "Number of hidden states for the HSMM")
        ("mindur", po::value<int>(), "Minimum duration of a segment")
        ("ndur", po::value<int>(), "Number of different durations supported")
        ("labels,l", po::value<vector<string>>(&input_labels)->multitoken(),
                "Path to input labels")
        ("nodur", "Flag to deactivate the learning of durations")
        ("alphadurprior", po::value<double>(),
                "Alpha for Dirichlet prior for the duration")
        ("filteringprediction", po::value<string>(), "Path to predicted labels"
                " based on the filtering distribution over states")
        ("mr", po::value<string>(), "Runlength marginals output filename")
        ("ms", po::value<string>(), "States marginals output filename")
        ("ms2", po::value<string>(), "States marginals output filename. This "
                "one is based on the residual time posterior instead of the "
                "runlength posterior")
        ("md", po::value<string>(), "Duration marginals output filename")
        ("ml", po::value<string>(), "Remaining runlength marginals output"
                " filename")
        ("leaveoneout", po::value<int>(), "Index of the sequence that will be"
                " left out for validation")
        ("savefiletype", po::value<string>()->default_value("arma_binary"),
                "File type to save the matrices after inference");
    assert(input_features.size() == input_labels.size());
    vector<string> required_fields = {"input", "labels", "leaveoneout",
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
    int nstates = vm["nstates"].as<int>();
    int ndurations = vm["ndur"].as<int>();
    int leaveoneout = vm["leaveoneout"].as<int>();
    int min_duration = vm["mindur"].as<int>();
    int nseq = input_features.size();

    field<ivec> labels_seq(nseq - 1);
    field<mat> obs_seq(nseq - 1);
    ivec test_labels;
    mat test_obs;
    int train_idx = 0;
    for(int i = 0; i < nseq; i++) {
        if (i == leaveoneout) {
            test_obs.load(input_features[i], raw_ascii);
            test_labels.load(input_labels[i], raw_ascii);
        }
        else {
            labels_seq(train_idx).load(input_labels[i], raw_ascii);
            obs_seq(train_idx).load(input_features[i], raw_ascii);
            train_idx++;
        }
    }
    assert(train_idx == nseq - 1);
    int ndimension = test_obs.n_rows;

    // Creating normal distributions for the emission process.
    vector<NormalDist> states = get_normal_distributions(nstates, ndurations,
            min_duration, ndimension);

    // Creating the emission process.
    shared_ptr<MultivariateGaussianEmission> emission(
            new MultivariateGaussianEmission(states));

    // Training the emission based on the labels.
    emission->fitFromLabels(obs_seq, labels_seq);

    // Creating the online HSMM whose emission process doesnt take into account
    // the total segment duration. The pmfs are uniformly initialized.
    OnlineHSMMRunlengthBased model(emission, nstates, ndurations, min_duration);

    // Setting a Dirichlet prior over the durations.
    if (vm.count("alphadurprior")) {
        mat alphas = ones<mat>(nstates, ndurations) *
            vm["alphadurprior"].as<double>();
        model.setDurationDirichletPrior(alphas);
    }

    // Learning the HSMM parameters from the labels.
    if (min_duration == 1 && ndurations == 1)
        model.setTransition(getHmmTransitionFromLabels(labels_seq, nstates));
    else
        model.setTransitionFromLabels(labels_seq);

    if (!vm.count("nodur"))
        model.setDurationFromLabels(labels_seq);

    if (vm.count("output")) {
        ofstream output_params(vm["output"].as<string>());
        nlohmann::json current_params = model.to_stream();
        output_params << std::setw(4) << current_params << endl;
        output_params.close();
    }
    mat runlength_marginals;
    mat state_marginals, state_marginals_2;
    mat remaining_runlength_marginals;
    mat duration_marginals;

    if (vm.count("mr"))
        runlength_marginals = zeros<mat>(min_duration + ndurations - 1,
                test_obs.n_cols);
    if (vm.count("ms") || vm.count("filteringprediction"))
        state_marginals = zeros<mat>(nstates, test_obs.n_cols);
    if (vm.count("ms2"))
        state_marginals_2 = zeros<mat>(nstates, test_obs.n_cols);
    if (vm.count("md"))
        duration_marginals = zeros<mat>(ndurations, test_obs.n_cols);
    if (vm.count("ml"))
        remaining_runlength_marginals = zeros<mat>(
                min_duration + ndurations - 1, test_obs.n_cols);
    vec loglikelihoods(test_obs.n_cols);
    for(int i = 0; i < test_obs.n_cols; i++) {
        loglikelihoods(i) = model.oneStepAheadLoglikelihood(test_obs.col(i));
        model.addNewObservation(test_obs.col(i));
        if (vm.count("mr"))
            runlength_marginals.col(i) = model.getRunlengthMarginal();
        if (vm.count("ms") || vm.count("filteringprediction"))
            state_marginals.col(i) = model.getStateMarginal();
        if (vm.count("ms2"))
            state_marginals_2.col(i) = model.getStateMarginal2();
        if (vm.count("ml"))
            remaining_runlength_marginals.col(i) =
                    model.getResidualTimeMarginal();
    }

    // The test log-likelihood.
    cout << accu(loglikelihoods) << endl;

    // Saving the filtering inferences.
    auto file_type = vm["savefiletype"].as<string>().compare(
            "arma_binary") == 0 ? arma_binary : raw_ascii;
    if (vm.count("mr"))
        runlength_marginals.save(vm["mr"].as<string>(), file_type);
    if (vm.count("ms"))
        state_marginals.save(vm["ms"].as<string>(), file_type);
    if (vm.count("ms2"))
        state_marginals_2.save(vm["ms2"].as<string>(), file_type);
    if (vm.count("ml"))
        remaining_runlength_marginals.save(vm["ml"].as<string>(), file_type);
    if (vm.count("filteringprediction")) {
        ivec filtering_prediction = predict_labels_from_filtering(
                state_marginals);
        filtering_prediction.save(vm["filteringprediction"].as<string>(),
                raw_ascii);
    }
    return 0;
}
