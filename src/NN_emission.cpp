#include <NN_emission.hpp>

using namespace arma;
using namespace std;


namespace hsmm {

    NNEmission* NNEmission::clone() const {
        return new NNEmission(*this);
    }

    double NNEmission::loglikelihood(int state, const field<mat>& obs) const {
        return 0; // TODO
    }

    void NNEmission::reestimate(int min_duration, const field<cube>& meta,
            const field<field<mat>>& mobs) {
        return; //TODO
    }

    field<mat> NNEmission::sampleFromState(int state, int size,
            mt19937 &rng) const {
        field<mat> ret;
        return ret; //TODO
    }

    field<mat> NNEmission::sampleNextObsGivenPastObs(int state, int seg_dur,
            const field<mat>& past_obs, mt19937 &rng) const {
        field<mat> ret;
        return ret; //TODO
    }

};
