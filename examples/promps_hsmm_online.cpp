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


int main(int argc, char *argv[]) {
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("params,p", po::value<string>(), "Path to the input promp hsmm params")
        ("polybasisfun", po::value<int>()->default_value(1), "Order of the "
                "poly basis functions")
        ("norbf", "Flag to deactivate the radial basis functions");
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
    auto comb = shared_ptr<ScalarCombBasis>(new ScalarCombBasis({rbf, poly}));
    if (vm.count("norbf"))
        comb = shared_ptr<ScalarCombBasis>(new ScalarCombBasis({poly}));
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
    HSMM promp_hsmm(std::static_pointer_cast<AbstractEmission>(ptr_emission),
            dummy_transition, dummy_pi, dummy_duration, min_duration);
    promp_hsmm.from_stream(current_params);
    return 0;
}

