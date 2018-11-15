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
        ("viterbi,v", po::value<string>(), "Path to the input viterbi file")
        ("nstates,s", po::value<int>(), "Number of states (NNs)")
        ("hiddenunits,u", po::value<int>()->default_value(10),
                "Number of hidden units")
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
    int hidden_units = vm["hiddenunits"].as<int>();
    vector<mat> obs_for_each_state[nstates];
    vector<mat> times_for_each_state[nstates];
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
            rowvec times = linspace<rowvec>(0, 1.0, dur(j));
            times_for_each_state[hs(j)].push_back(times);
            idx += dur(j);
        }
    }

    // There should be at least one segment for the hidden state 0.
    int njoints = obs_for_each_state[0].at(0).n_rows;
    FFN<MeanSquaredError<>, RandomInitialization> neural_network[nstates];
    for(int i = 0; i < nstates; i++) {
        mat inputs = join_mats(times_for_each_state[i]);
        mat outputs = join_mats(obs_for_each_state[i]);
        assert (outputs.n_rows == njoints);
        assert(outputs.n_cols == inputs.n_cols);

        // Defining the architecture of the NN.
        neural_network[i].Add<Linear<>>(1, hidden_units);
        neural_network[i].Add<SigmoidLayer<>>();
        neural_network[i].Add<Linear<>>(hidden_units, njoints);

        // Training the NN.
        neural_network[i].Train(inputs, outputs);

        // Evaluating the loss.
        mat test_input = linspace<rowvec>(0,1,100);
        mat test_output;
        neural_network[i].Predict(test_input, test_output);
    }
    return 0;
}
