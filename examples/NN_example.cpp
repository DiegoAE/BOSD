#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <armadillo>
#include <json.hpp>
#include <HSMM.hpp>
#include <NN_emission.hpp>

using namespace arma;
using namespace mlpack::ann;
using namespace hsmm;
using namespace std;
using json = nlohmann::json;

int main() {
    int min_duration = 10;
    int ndurations = 10;
    int nstates = 3;
    int njoints = 3;

    mat transition(nstates, nstates);
    transition.fill(1.0 / nstates );
    vec pi(nstates);
    pi.fill(1.0/nstates);
    mat durations(nstates, ndurations);
    durations.fill(1.0 / ndurations);

    std::vector<NNmodel> ffns;
    for(int i = 0; i < 3; i++) {
        NNmodel model;
        model.Add<Linear<> >(1, 10);
        model.Add<SigmoidLayer<> >();
        model.Add<Linear<> >(10, njoints);
        ffns.push_back(model);
    }

    // Creating the NN emission.
    shared_ptr<NNEmission> ptr_emission(new NNEmission(ffns, njoints));

    OnlineHSMM online_promp_hsmm(std::static_pointer_cast<
            AbstractEmissionOnlineSetting>(ptr_emission), transition, pi,
            durations, min_duration);
    json params = online_promp_hsmm.to_stream();
    cout << params.dump(4) << endl;
    return 0;
}

