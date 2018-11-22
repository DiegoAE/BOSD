#include <iostream>
#include <NN_emission.hpp>
#include <armadillo>
#include <NN_basis_function.hpp>
#include <json.hpp>
#include <HSMM.hpp>

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

    // Creating the NN emission.
    shared_ptr<NNEmission> ptr_emission(new NNEmission(nstates, njoints));

    OnlineHSMM online_promp_hsmm(std::static_pointer_cast<
            AbstractEmissionOnlineSetting>(ptr_emission), transition, pi,
            durations, min_duration);
    json params = online_promp_hsmm.to_stream();
    cout << params.dump(4) << endl;

    // TODO.
    //ivec hidden_states, hidden_durations;
    //field<mat> seq = online_promp_hsmm.sampleSegments(1, hidden_states,
    //        hidden_durations);
    ScalarNNBasis prueba(10);
    return 0;
}

