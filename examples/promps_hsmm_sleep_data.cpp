#include <armadillo>
#include <boost/program_options.hpp>
#include <Multivariate_Gaussian_emission.hpp>
#include <random>

using namespace arma;
using namespace hsmm;
using namespace robotics::random;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    mt19937 gen(0);
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("eeg1", po::value<string>(), "Path to input obs")
        ("eeg2", po::value<string>(), "Path to input obs")
        ("emg", po::value<string>(), "Path to input obs")
        ("labels,l", po::value<string>(), "Path to input labels");
    vector<string> required_fields = {"eeg1", "eeg2", "emg", "labels"};
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

    // Reading inputs and labels.
    mat eeg1, eeg2, emg;
    eeg1.load(vm["eeg1"].as<string>(), raw_ascii);
    eeg2.load(vm["eeg2"].as<string>(), raw_ascii);
    emg.load(vm["emg"].as<string>(), raw_ascii);
    ivec gt_labels;
    gt_labels.load(vm["labels"].as<string>(), raw_ascii);

    int nstates = 5;
    int ndimension = 10;
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
    MultivariateGaussianEmission emission(states);
    ivec labels_vec = conv_to<ivec>::from(labels);
    field<vec> obs_field(samples.size());
    for(int i = 0; i < samples.size(); i++)
        obs_field(i) = samples.at(i);

    // TODO: plug in the inputs and labels.
    emission.fitFromLabels(obs_field, labels_vec);
    return 0;
}
