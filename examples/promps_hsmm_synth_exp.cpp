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
        ("vit,v", po::value<string>(), "Filename to store the viterbi output")
        ("ms", po::value<string>(), "state marginals file name")
        ("mr", po::value<string>(), "runlength marginals file name")
        ("md", po::value<string>(), "duration marginals file name ")
        ("imd", po::value<string>(), "implicit duration marginals file name."
                " This means it is computed from the runlength and state")
        ("delta", po::value<double>(), "delta between sample locations");
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
    int njoints = 1;

    // Setting a combination of polynomial and rbf basis functions.
    auto rbf = shared_ptr<ScalarGaussBasis>(new ScalarGaussBasis(
                {0.0,0.2,0.4,0.6,0.8,1.0}, 0.25));
    auto poly = make_shared<ScalarPolyBasis>(1);
    auto comb = shared_ptr<ScalarCombBasis>(new ScalarCombBasis({rbf, poly}));
    int n_basis_functions = comb->dim();
    int nparameters = n_basis_functions * njoints;

    mat promp_means = {{0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}};
    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w = conv_to<vec>::from(promp_means.row(i));
        mat Sigma_w = eye<mat>(nparameters, nparameters);
        mat Sigma_y = 0.00001*eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(comb, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission.
    shared_ptr<AbstractEmissionOnlineSetting> ptr_emission(new ProMPsEmission(
                promps));

    OnlineHSMM promp_hsmm(ptr_emission, transition, pi, durations,
            min_duration);

    int nseq = 1;
    int nsegments = 10;
    field<ivec> hidden_states, hidden_durations;
    field<field<mat>> multiple_toy_obs = promp_hsmm.sampleMultipleSequences(
            nseq, nsegments, hidden_states, hidden_durations);
    cout << "Generated states and durations for the first sequence" << endl;
    imat viterbi = join_horiz(hidden_states(0), hidden_durations(0));
    cout << viterbi << endl;

    cout << "Model Parameters" << endl;
    json params = promp_hsmm.to_stream();
    cout << params.dump(4) << endl;

    mat obs = fieldToMat(njoints, multiple_toy_obs(0));
    if (vm.count("output"))
        obs.save(vm["output"].as<string>(), raw_ascii);
    if (vm.count("vit"))
        viterbi.save(vm["vit"].as<string>(), raw_ascii);

    // Note that delta only affects the online filtering, not the generation.
    if (vm.count("delta"))
        std::static_pointer_cast<ProMPsEmission>(
                ptr_emission)->setDelta(vm["delta"].as<double>());

    mat state_marginals_over_time(nstates, obs.n_cols);
    mat runlength_marginals_over_time(min_duration + ndurations,
            obs.n_cols);
    mat duration_marginals_over_time(ndurations, obs.n_cols);
    mat implicit_duration_marginals_over_time(ndurations, obs.n_cols);
    for(int c = 0; c < obs.n_cols; c++) {
        promp_hsmm.addNewObservation(obs.col(c));
        vec s_marginal = promp_hsmm.getStateMarginal();
        state_marginals_over_time.col(c) = s_marginal;
        vec r_marginal = promp_hsmm.getRunlengthMarginal();
        runlength_marginals_over_time.col(c) = r_marginal;
        vec d_marginal = promp_hsmm.getDurationMarginal();
        duration_marginals_over_time.col(c) = d_marginal;
        if (vm.count("imd")) {
            vec indirect_d_marginal = promp_hsmm.getImplicitDurationMarginal();
            implicit_duration_marginals_over_time.col(c) = indirect_d_marginal;
        }
    }

    // Saving the marginals if required.
    if (vm.count("ms"))
        state_marginals_over_time.save(vm["ms"].as<string>(), raw_ascii);
    if (vm.count("mr"))
        runlength_marginals_over_time.save(vm["mr"].as<string>(), raw_ascii);
    if (vm.count("md"))
        duration_marginals_over_time.save(vm["md"].as<string>(), raw_ascii);
    if (vm.count("imd"))
        implicit_duration_marginals_over_time.save(vm["imd"].as<string>(),
                raw_ascii);
    return 0;
}
