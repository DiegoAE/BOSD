#include <armadillo>
#include <cmath>
#include <json.hpp>
#include <iostream>
#include <Multivariate_Gaussian_emission.hpp>

using namespace arma;
using namespace robotics;
using namespace std;
using json = nlohmann::json;   
 
namespace hsmm {

    /**
     * MultivariateGaussianEmission implementation.
     */
    MultivariateGaussianEmission::MultivariateGaussianEmission(
            vector<random::NormalDist> states) : states_(states),
            AbstractEmissionObsCondIIDgivenState(states.size(),
            states.at(0).mean().n_elem) {
        for(auto& dist: states_)
            assert(dist.mean().n_elem == getDimension());
    }

    MultivariateGaussianEmission* MultivariateGaussianEmission::clone() const {
        return new MultivariateGaussianEmission(*this);
    }

    double MultivariateGaussianEmission::loglikelihood(int state,
            const vec &single_obs) const {
        return random::log_normal_density(states_.at(state), single_obs);
    }

    void MultivariateGaussianEmission::fitFromLabels(
            const field<vec> &observations, const ivec &labels) {
        assert(observations.n_elem == labels.n_elem);
        vector<vec> obs_for_each_state[states_.size()];
        for(int i = 0; i < labels.n_elem; i++)
            obs_for_each_state[labels(i)].push_back(conv_to<vec>::from(
                        observations(i)));
        for(int i = 0; i < states_.size(); i++)
            states_.at(i) = random::mle_multivariate_normal(
                    obs_for_each_state[i]);
    }

    field<mat> MultivariateGaussianEmission::sampleFromState(int state,
            int size, mt19937 &rng) const {
        vector<vec> s = sample_multivariate_normal(rng, states_.at(state),
                size);
        field<mat> ret(size);
        for(int i = 0; i < size; i++)
            ret(i) = s.at(i);
        return ret;
    }
};

