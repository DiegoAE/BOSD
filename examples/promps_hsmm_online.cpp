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

mat fieldToMat(int njoints, field<mat> &samples) {
    mat ret(njoints, samples.n_elem);
    for(int i = 0; i < samples.n_elem; i++)
        ret.col(i) = samples(i);
    return ret;
}

int main(int argc, char *argv[]) {
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("params,p", po::value<string>(), "Path to the input promp hsmm params")
        ("input,i", po::value<string>(), "Path to input obs. (optional)")
        ("upto,u", po::value<int>(), "The model will condition on the obs. up "
                "to this index")
        ("nsamples,n", po::value<int>()->default_value(1), "Number of samples "
                "to generate after conditioning on the input observations")
        ("ntimeseries,t", po::value<int>()->default_value(1), "Number of i.i.d."
                " time series to generate conditioned on the same obs")
        ("output,o", po::value<string>(), "Filename of the output samples")
        ("polybasisfun", po::value<int>()->default_value(2), "Order of the "
                "poly basis functions")
        ("rbf", "Flag to activate the radial basis functions");
    vector<string> required_fields = {"params"};
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
    // Reading the current parameters.
    string input_filename = vm["params"].as<string>();
    std::ifstream current_params_stream(input_filename);
    nlohmann::json current_params;
    current_params_stream >> current_params;
    int nstates = current_params["nstates"];
    int nduration = current_params["ndurations"];
    int njoints = current_params["emission_params"][0]["num_joints"];
    int min_duration = current_params["min_duration"];

    // Setting a combination of polynomial and rbf basis functions.
    auto rbf = shared_ptr<ScalarGaussBasis>(new ScalarGaussBasis(
                {0.25,0.5,0.75},0.25));
    auto poly = make_shared<ScalarPolyBasis>(vm["polybasisfun"].as<int>());
    auto comb = shared_ptr<ScalarCombBasis>(new ScalarCombBasis({poly}));
    if (vm.count("rbf"))
        comb = shared_ptr<ScalarCombBasis>(new ScalarCombBasis({rbf, poly}));
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

    mat dummy_transition(nstates, nstates, fill::eye);
    vec dummy_pi(nstates, fill::eye);
    mat dummy_duration(nstates, nduration, fill::eye);

    // Creating the ProMP emission.
    shared_ptr<ProMPsEmission> ptr_emission(new ProMPsEmission(promps));
    OnlineHSMM online_promp_hsmm(std::static_pointer_cast<
            AbstractEmissionOnlineSetting>(ptr_emission), dummy_transition,
            dummy_pi, dummy_duration,
            min_duration);
    online_promp_hsmm.from_stream(current_params);
    online_promp_hsmm.debug_ = true;
    mat obs;
    if (vm.count("input")) {
        obs.load(vm["input"].as<string>(), raw_ascii);
        assert(obs.n_rows == njoints);
    }
    mat obs_for_cond(obs);
    if (!obs.is_empty() && vm.count("upto"))
        obs_for_cond = obs.cols(0, vm["upto"].as<int>());

    for(int c = 0; c < obs_for_cond.n_cols; c++)
        online_promp_hsmm.addNewObservation(obs_for_cond.col(c));

    online_promp_hsmm.printTopKFromPosterior(15);

    int nsamples = vm["nsamples"].as<int>();

    for(int j = 0; j < vm["ntimeseries"].as<int>(); j++) {
        field<mat> fsamples = online_promp_hsmm.sampleNextObservations(nsamples);
        mat samples(njoints, fsamples.n_elem);
        for(int c = 0; c < fsamples.n_elem; c++)
            samples.col(c) = fsamples(c);
        mat whole_ts = join_horiz(obs_for_cond, samples);
        if (vm.count("output")) {
            string filename = vm["output"].as<string>() + "." + to_string(j);
            whole_ts.save(filename, raw_ascii);
        }
        else
            whole_ts.raw_print(cout);
    }
    return 0;
}

