#include <NN_emission.hpp>

using namespace arma;
using namespace mlpack::ann;
using namespace std;


namespace hsmm {

    NNEmission::NNEmission(int nstates, int njoints) :
            AbstractEmissionOnlineSetting(nstates, njoints),
            noise_var_(njoints, nstates, arma::fill::ones) {
        for(int i = 0; i < nstates; i++) {
            NNmodel model;
            model.Add<Linear<> >(1, 10);
            model.Add<SigmoidLayer<> >();
            model.Add<Linear<> >(10, njoints);
            ffns_.push_back(model);
        }
    }

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
        //vec sample_locations = getSampleLocations(seg_dur);
        //mat sample;
        //ffns_.at(state).Predict(sample_locations, sample);
        //cout << sample << endl;
        field<mat> ret;
        return ret; //TODO
    }

    vec NNEmission::getSampleLocations(int length) const {
        if (sample_locations_delta_ < 0)
            return linspace<vec>(0, 1.0, length);
        else
            return linspace<vec>(0, (length-1)*sample_locations_delta_,
                    length);
    }

};
