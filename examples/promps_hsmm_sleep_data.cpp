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
    desc.add_options()
        ("help,h", "Produce help message")
        ("output,o", po::value<string>(), "Path to the json output params")
        ("nstates", po::value<int>(), "Number of hidden states for the HSMM")
        ("mindur", po::value<int>(), "Minimum duration of a segment")
        ("ndur", po::value<int>(), "Number of different durations supported")
        ("eeg1", po::value<string>(), "Path to input obs")
        ("eeg2", po::value<string>(), "Path to input obs")
        ("emg", po::value<string>(), "Path to input obs")
        ("labels,l", po::value<string>(), "Path to input labels")
        ("alphadurprior", po::value<int>(),
                "Alpha for Dirichlet prior for the duration")
        ("iidprediction", po::value<string>(), "Path to predicted labels based"
                " on the iid assumption across epochs")
        ("filteringprediction", po::value<string>(), "Path to predicted labels"
                " based on the filtering distribution over states")
        ("groundtruth,g", po::value<string>(), "Path to where the true labels"
                " are stored")
        ("mr", po::value<string>(), "Runlength marginals output filename")
        ("ms", po::value<string>(), "States marginals output filename");
    vector<string> required_fields = {"eeg1", "eeg2", "emg", "labels",
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
    int min_duration = vm["mindur"].as<int>();
    int ndimension = 7;

    // Creating normal distributions for the emission process.
    vector<NormalDist> states = get_normal_distributions(nstates, ndurations,
            min_duration, ndimension);

    // Reading inputs and labels.
    mat eeg1, eeg2, emg;
    ivec gt_labels;
    eeg1.load(vm["eeg1"].as<string>(), raw_ascii);
    eeg2.load(vm["eeg2"].as<string>(), raw_ascii);
    emg.load(vm["emg"].as<string>(), raw_ascii);
    gt_labels.load(vm["labels"].as<string>(), raw_ascii);

    // Computing the FFT features.
    field<vec> features = getFeatureVectors(eeg1, eeg2, emg);
    const field<vec>& train_features = features.rows(0, 21600 * 2 - 1);
    const ivec& train_labels = gt_labels.head(21600 * 2);
    const field<vec>& test_features = features.rows(21600 * 2,
            features.n_elem - 1);
    const ivec& test_labels = gt_labels.tail(21600);

    assert(features.at(0).n_elem == ndimension);

    // Creating the emission process.
    shared_ptr<MultivariateGaussianEmission> emission(
            new MultivariateGaussianEmission(states));

    // Training the emission based on the labels.
    emission->fitFromLabels(train_features, train_labels);

    // Defining the prior over hidden states.
    vec labels_prior(nstates, fill::zeros);
    for(int i = 0; i < train_labels.n_elem; i++)
        labels_prior(train_labels(i))++;
    labels_prior = labels_prior / accu(labels_prior);

    // Setting uniform prior.
    // labels_prior = ones<vec>(nstates) / nstates;

    // IID prediction.
    if (vm.count("iidprediction")) {
        ivec prediction = predict_labels_iid(emission, test_features,
                labels_prior);
        prediction.save(vm["iidprediction"].as<string>(), raw_ascii);
    }

    if (vm.count("groundtruth"))
        test_labels.save(vm["groundtruth"].as<string>(), raw_ascii);

    // Creating the online HSMM whose emission process doesnt take into account
    // the total segment duration. The pmfs are uniformly initialized.
    OnlineHSMMRunlengthBased model(emission, nstates, ndurations, min_duration);

    // Setting a Dirichlet prior over the durations.
    if (vm.count("alphadurprior")) {
        mat alphas = ones<mat>(nstates, ndurations) *
            vm["alphadurprior"].as<int>();
        model.setDurationDirichletPrior(alphas);
    }

    // Learning the HSMM parameters from the labels.
    field<ivec> training_labels_seqs = {train_labels};
    model.setTransitionFromLabels(training_labels_seqs);
    model.setDurationFromLabels(training_labels_seqs);

    if (vm.count("output")) {
        ofstream output_params(vm["output"].as<string>());
        nlohmann::json current_params = model.to_stream();
        output_params << std::setw(4) << current_params << endl;
        output_params.close();
    }

    mat runlength_marginals(min_duration + ndurations, test_features.n_elem);
    mat state_marginals(nstates, test_features.n_elem);
    for(int i = 0; i < test_features.n_elem; i++) {
        model.addNewObservation(test_features.at(i));
        runlength_marginals.col(i) = model.getRunlengthMarginal();
        state_marginals.col(i) = model.getStateMarginal();
    }
    if (vm.count("mr"))
        runlength_marginals.save(vm["mr"].as<string>(), raw_ascii);
    if (vm.count("ms"))
        state_marginals.save(vm["ms"].as<string>(), raw_ascii);
    if (vm.count("filteringprediction")) {
        ivec filtering_prediction = predict_labels_from_filtering(
                state_marginals);
        filtering_prediction.save(vm["filteringprediction"].as<string>(),
                raw_ascii);
    }
    return 0;
}
