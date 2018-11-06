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
        ("params,p", po::value<string>(), "JSON input params (optional)")
        ("output,o", po::value<string>(), "Filename to store the obs")
        ("vit,v", po::value<string>(), "Filename to store the viterbi output")
        ("ms", po::value<string>(), "state marginals file name")
        ("mr", po::value<string>(), "runlength marginals file name")
        ("md", po::value<string>(), "duration marginals file name ")
        ("imd", po::value<string>(), "implicit duration marginals file name."
                " This means it is computed from the runlength and state")
        ("polybasisfun", po::value<int>()->default_value(1), "Order of the"
                " poly basis")
        ("rbfbasisfun", po::value<int>()->default_value(3), "Number of radial"
                " basis functions to use between 0 and 1. 0,1 are excluded.")
        ("delta", po::value<double>(), "delta between sample locations")
        ("print_ll_vit", "If set then the ll from each hs for"
                " every ground truth segment is printed. Intended for "
                "debugging purposes");
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
    json input_params;
    if (vm.count("params")) {
        string params = vm["params"].as<string>();
        ifstream input_params_file(params);
        input_params_file >> input_params;
        min_duration = input_params["min_duration"];
        nstates = input_params["nstates"];
        ndurations = input_params["ndurations"];
        transition = eye<mat>(nstates, nstates);
        pi = eye<vec>(nstates, 1);
        durations = eye<mat>(nstates, ndurations);
        njoints = input_params["emission_params"][0]["num_joints"];
    }

    // Setting a combination of polynomial and rbf basis functions.
    auto poly = make_shared<ScalarPolyBasis>(vm["polybasisfun"].as<int>());
    auto comb = shared_ptr<ScalarCombBasis>(new ScalarCombBasis({poly}));
    int nrbf = vm["rbfbasisfun"].as<int>();
    if (nrbf > 0) {
        vec centers = linspace<vec>(0, 1.0, nrbf + 2);
        centers = centers.subvec(1, centers.n_elem - 2);
        auto rbf = shared_ptr<ScalarGaussBasis>(new ScalarGaussBasis(centers,
                    0.25));
        comb = shared_ptr<ScalarCombBasis>(new ScalarCombBasis({rbf, poly}));
    }
    int n_basis_functions = comb->dim();
    int nparameters = n_basis_functions * njoints;

    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w = zeros<vec>(nparameters);
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
    if (vm.count("params"))
        promp_hsmm.from_stream(input_params);

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

    if (vm.count("print_ll_vit")) {
        int idx = 0;
        for(int i = 0; i < viterbi.n_rows; i++) {
            int dur = viterbi(i, 1);
            int hs = viterbi(i, 0);
            cout << "Segment #" << i << ": (" << hs << "," << dur << ")" <<
                    endl;
            const field<mat>& segment = multiple_toy_obs(0).rows(idx,
                    idx + dur - 1);
            idx += dur;
            for(int i = 0; i < nstates; i++) {
                double ll = promp_hsmm.emission_->loglikelihood(i, segment);
                double lld = ll + promp_hsmm.duration_(i, dur - min_duration);
                cout << "State " << i << ": " << ll << " with dur: " << lld <<
                        endl;
            }
        }
    }

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
        if (vm.count("print_ll_vit")) {
            cout << "Obs idx: " << c << endl;
            promp_hsmm.printTopKFromPosterior(40);
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
