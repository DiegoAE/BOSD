#ifndef FFT_FEATURES_EMISSION_H
#define FFT_FEATURES_EMISSION_H

#include <armadillo>
#include <emissions.hpp>
#include <iostream>
#include <robotics/utils/random.hpp>
#include <vector>

namespace hsmm {

    class MultivariateGaussianEmission : public
                                         AbstractEmissionObsCondIIDgivenState {
        public:
            MultivariateGaussianEmission(
                    std::vector<robotics::random::NormalDist> states_);

            double loglikelihood(int state,
                    const arma::mat &single_obs) const;

            void fitFromLabels(const arma::field<arma::vec> &observations,
                    const arma::ivec &labels);

        private:
            std::vector<robotics::random::NormalDist> states_;
    };

};

#endif

