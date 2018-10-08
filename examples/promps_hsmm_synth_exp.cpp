#include <armadillo>
#include <boost/program_options.hpp>
#include <HSMM.hpp>
#include <iostream>
#include <json.hpp>
#include <memory>
#include <ProMPs_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace std;
using json = nlohmann::json;
namespace po = boost::program_options;


mat fieldToMat(int njoints, field<mat> &samples) {
    mat ret(njoints, samples.n_elem);
    for(int i = 0; i < samples.n_elem; i++)
        ret.col(i) = samples(i);
    return ret;
}

vec pmfFromGaussian(double mean, double var, int size, int min_duration) {
    vec pmf(size);
    for (int t = min_duration; t < min_duration + size; t++) {
        int idx = t - min_duration;
        double tmp = ((t - mean) * (t - mean)) / var;
        pmf(idx) = exp(-0.5 * tmp);
    }
    pmf = pmf * (1.0 / sum(pmf));
    return pmf;
}

int main(int argc, char *argv[]) {
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("output,o", po::value<string>(), "Filename to store the obs")
        ("vit,v", po::value<string>(), "Filename to store the viterbi output");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }
    int min_duration = 30;
    mat transition = ones<mat>(2, 2);
    transition.diag().zeros();
    int nstates = transition.n_rows;
    int ndurations = 30;
    vec pi = {0.5, 0.5};
    mat durations(nstates, ndurations);
    durations.row(0) = conv_to<rowvec>::from(pmfFromGaussian(
                40, 16, ndurations, min_duration));
    durations.row(1) = conv_to<rowvec>::from(pmfFromGaussian(
                50, 16, ndurations, min_duration));
    int n_basis_functions = 4;
    int njoints = 1;
    int nparameters = n_basis_functions * njoints;

    // Setting a third order polynomial basis function for the ProMP
    int polynomial_order = n_basis_functions - 1;
    shared_ptr<ScalarBasisFun> kernel{ new ScalarPolyBasis(polynomial_order)};

    mat promp_means = {{1,0,2,0.5}, {0,1,0,3}};
    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w = conv_to<vec>::from(promp_means.row(i));
        mat Sigma_w = eye<mat>(nparameters, nparameters);
        mat Sigma_y = 0.00001*eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(kernel, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission.
    shared_ptr<AbstractEmission> ptr_emission(new ProMPsEmission(promps));

    HSMM promp_hsmm(ptr_emission, transition, pi, durations, min_duration);

    int nseq = 1;
    int nsegments = 5;
    field<ivec> hidden_states, hidden_durations;
    field<field<mat>> multiple_toy_obs = promp_hsmm.sampleMultipleSequences(
            nseq, nsegments, hidden_states, hidden_durations);
    cout << "Generated states and durations for the first sequence" << endl;
    imat viterbi = join_horiz(hidden_states(0), hidden_durations(0));
    cout << viterbi << endl;

    cout << "Model Parameters" << endl;
    json params = promp_hsmm.to_stream();
    cout << params.dump(4) << endl;

    if (vm.count("output")) {
        mat obs = fieldToMat(njoints, multiple_toy_obs(0));
        obs.save(vm["output"].as<string>(), raw_ascii);
    }
    if (vm.count("vit"))
        viterbi.save(vm["vit"].as<string>(), raw_ascii);
    return 0;
}
