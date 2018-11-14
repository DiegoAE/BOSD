#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <armadillo>
#include <boost/program_options.hpp>
#include <iostream>

using namespace arma;
using namespace mlpack::ann;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("input,i", po::value<string>(), "Path to the input obs")
        ("viterbi,v", po::value<string>(), "Path to the input viterbi file")
        ("nstates,s", po::value<int>(), "Number of states (NNs)")
        ("nfiles,n", po::value<int>()->default_value(1),
                "Number of input files to process");
    vector<string> required_fields = {"input", "viterbi", "nfiles",
            "nstates"};
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
    string viterbi_filename = vm["viterbi"].as<string>();
    int nseq = vm["nfiles"].as<int>();
    int nstates = vm["nstates"].as<int>();
    vector<mat> obs_for_each_state[nstates];
    // TODO: generate the time indexes.
    for(int i = 0; i < nseq; i++) {
        string iname = input_filename;
        string vname = viterbi_filename;
        if (nseq != 1) {
            iname += string(".") + to_string(i);
            vname += string(".") + to_string(i);
        }
        mat obs;
        obs.load(iname, raw_ascii);
        imat vit;
        vit.load(vname, raw_ascii);
        ivec hs = vit.col(0);
        ivec dur = vit.col(1);
        int idx = 0;
        for(int j = 0; j < dur.n_rows; j++) {
            mat segment = obs.cols(idx, idx + dur(j) - 1);
            obs_for_each_state[hs(j)].push_back(segment);
            idx += dur(j);
        }
    }
    for(int i = 0; i < nstates; i++)
        cout << obs_for_each_state[i].size() << endl;
    int njoints = 3;
    int hidden_units = 10;
    FFN<MeanSquaredError<>, RandomInitialization> neural_network;
    neural_network.Add<Linear<>>(1, hidden_units);
    neural_network.Add<SigmoidLayer<>>();
    neural_network.Add<Linear<>>(hidden_units, njoints);
    cout << "OK" << endl;
    return 0;
}

