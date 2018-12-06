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
        ("durmomentmatching", "Flag to active the Gaussian moment matching"
                " for the duration learning")
        ("polybasisfun", po::value<int>()->default_value(0), "Order of the "
                "poly basis functions")
        ("noselftransitions", "Flag to deactive self transitions")
	("delta", po::value<double>(), "If this is given a value, the model "
                "switches to segment agnostic and the samples are generated "
                "according to the provided delta")
        ("savebasisfunparams", po::value<string>(), "File where the NN weights"
                " will be saved after training");
    vector<string> required_fields = {"input", "output", "viterbilabels",
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
    string output_filename = vm["output"].as<string>();
    string viterbi_filename = vm["viterbilabels"].as<string>();
    int nseq = vm["nfiles"].as<int>();
    int nstates = vm["nstates"].as<int>();
    int hidden_units = vm["hiddenunits"].as<int>();
    int nlayers = vm["nlayers"].as<int>();
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

            // TODO: take into account delta. Normalizing the time for now.
            rowvec times = linspace<rowvec>(0, 1.0, dur(j));
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
        auto nn1 = make_shared<ScalarNNBasis>(hidden_units_per_layer, njoints);

        // Training the NN.
        nn1->getNeuralNet().Train(inputs, outputs);
        auto serialized = nn1->to_stream();
        basis_fun_params.push_back(serialized);

        // Testing the NN building from serialized parameters.
        auto nn = make_shared<ScalarNNBasis>(serialized);

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
            vec output2 = nn1->eval(test_input(j));
            assert(approx_equal(output, output2, "reldiff", 1e-8));
            test_output.push_back(output);
        }
        mat mat_test_output = join_mats(test_output);
        //mat_test_output.save("prediction.txt." + to_string(i) , raw_ascii);
        basis.push_back(comb);
    }
    int n_basis_functions = basis.at(0)->dim();

    if (vm.count("savebasisfunparams")) {
        nlohmann::json basisfunparams = basis_fun_params;
        std::ofstream basis_fun_file(vm["savebasisfunparams"].as<string>());
        basis_fun_file << std::setw(4) << basisfunparams << std::endl;
        basis_fun_file.close();
    }
    return 0;
}

