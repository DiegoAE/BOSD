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
            assert(dist.mean().n_elem == dimension_);
    }

    double MultivariateGaussianEmission::loglikelihood(int state,
            const mat &single_obs) const {
        //TODO
        return 0.0;
    }

    void MultivariateGaussianEmission::fitFromLabels(field<mat> &observations,
            ivec &labels) {
        // TODO
        return;
    }
};

