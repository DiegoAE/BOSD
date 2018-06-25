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
        ("input,i", po::value<string>(), "Path to the input obs")
        ("output,o", po::value<string>(), "Path to the json output params")
        ("commitid", po::value<string>(), "Git commit id of the experiment")
        ("labels", po::value<string>(), "Path to the provided labels")
        ("nfiles", po::value<int>()->default_value(1),
                "Number of files (sequences) to process");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }
    if (!vm.count("input") || !vm.count("output")) {
        cerr << "Error: You should provide input and output files" << endl;
        return 1;
    }
    string input_filename = vm["input"].as<string>();
    string output_filename = vm["output"].as<string>();
    field<mat> seq_obs(1);
    field<Labels> seq_labels(1);
    int njoints;
    int nseq = vm["nfiles"].as<int>();
    seq_obs.set_size(nseq);
    seq_labels.set_size(nseq);
    for(int i = 0; i < seq_obs.n_elem; i++) {
        string name = input_filename;
        if (nseq != 1)
            name += string(".") + to_string(i);
        mat obs;
        obs.load(name, raw_ascii);
        njoints = obs.n_rows;
        int nobs = obs.n_cols;
        cout << "Time series shape: (" << njoints << ", " << nobs <<
            ")." << endl;
        seq_obs(i) = obs;

        // Reading labels for different obs.
        if (!vm.count("labels"))
            continue;
        string labels_name = vm["labels"].as<string>();
        if (nseq != 1)
            labels_name += string(".") + to_string(i);
        mat labels_mat;
        labels_mat.load(labels_name);
        for(int j = 0; j < labels_mat.n_rows; j++)
            seq_labels(i).setLabel(labels_mat(j, 0), labels_mat(j, 1),
                    labels_mat(j, 2));
    }
    int min_duration = 35;
    int nstates = 5;
    int ndurations = 100;
    mat transition(nstates, nstates);
    transition.fill(1.0 / nstates );
    vec pi(nstates);
    pi.fill(1.0/nstates);
    mat durations(nstates, ndurations);
    durations.fill(1.0 / ndurations);
    int n_basis_functions = 4;

    // Setting a third order polynomial basis function for the ProMP
    int polynomial_order = n_basis_functions - 1;
    shared_ptr<ScalarBasisFun> kernel{ new ScalarPolyBasis(polynomial_order)};

    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w(n_basis_functions * njoints, fill::randn);
        mat Sigma_w = 10 * eye<mat>(n_basis_functions * njoints,
                    n_basis_functions * njoints);
        mat Sigma_y = 0.01 * eye<mat>(njoints, njoints);

        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(kernel, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission.
    shared_ptr<ProMPsEmission> ptr_emission(new ProMPsEmission(promps));

    // Creating a prior for Sigma_w.
    mat Phi = 0.01 * eye<mat>(n_basis_functions * njoints,
            n_basis_functions * njoints);
    InverseWishart iw_prior(Phi, Phi.n_rows + 2);
    ptr_emission->set_Sigma_w_Prior(iw_prior);

    HSMM promp_hsmm(std::static_pointer_cast<AbstractEmission>(ptr_emission),
            transition, pi, durations, min_duration);

    // Saving the model in a json file.
    std::ofstream initial_params(output_filename);
    nlohmann::json initial_model = promp_hsmm.to_stream();
    if (vm.count("commitid"))
        initial_model["git_commit_id"] = vm["commitid"].as<string>();
    initial_params << std::setw(4) << initial_model << std::endl;
    initial_params.close();

    for(int i = 0; i < 10; i++) {

        // Reading the current parameters.
        std::ifstream current_params_stream(output_filename);
        nlohmann::json current_params;
        current_params_stream >> current_params;
        promp_hsmm.from_stream(current_params);
        bool convergence_reached = promp_hsmm.fit(seq_obs, seq_labels, 5, 1e-5);

        // Saving again the parameters after one training iteration.
        std::ofstream output_params(output_filename);
        current_params = promp_hsmm.to_stream();
        if (vm.count("commitid"))
            current_params["git_commit_id"] = vm["commitid"].as<string>();
        output_params << std::setw(4) << current_params << std::endl;
        output_params.close();

        if (convergence_reached)
            break;

    }
    return 0;
}
