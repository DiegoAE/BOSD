#include <NN_emission.hpp>

using namespace arma;
using namespace mlpack::ann;
using namespace std;


namespace hsmm {

    double zeroMeanGaussianLogLikelihoodR(const vec& x, const mat &precision) {
        return -0.5 * as_scalar(x.t()*precision*x - log(det(precision)) +
                x.n_elem * log(2*datum::pi));
    }

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

        // The samples are assumed to be equally spaced.
        vec sample_locations = getSampleLocations(obs.n_elem);

        // Noise precision.
        mat precision  = diagmat(1. / noise_var_.col(state));

        double ret = 0;
        for(int i = 0; i < obs.n_elem; i++) {

            if (obs(i).is_empty()) {

                // Making sure all the missing obs are at the end.
                // Other missing obs patterns are not supported yet.
                for(int j = i; j < obs.n_elem; j++)
                    assert(obs(j).is_empty());
                break;
            }
            mat output;
            mat input = {sample_locations(i)};
            ffns_.at(state).Predict(input, output);
            vec diff = obs(i) - output;
            ret += zeroMeanGaussianLogLikelihoodR(diff, precision);
        }
        return ret;
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

    NNmodel& NNEmission::getNeuralNet(int state) {
        return ffns_.at(state);
    }

    void NNEmission::setNoiseVar(const mat& noise_var) {
        assert(noise_var.n_rows == getDimension() &&
                noise_var.n_cols == getNumberStates());
        noise_var_ = noise_var;
    }

};
