#ifndef FFT_FEATURES_EMISSION_H
#define FFT_FEATURES_EMISSION_H

#include <armadillo>
#include <emissions.hpp>
#include <iostream>
#include <robotics/utils/random.hpp>

using namespace arma;
using namespace std;

namespace hsmm {

    class MultivariateGaussianEmission : public
                                         AbstractEmissionObsCondIIDgivenState {
        public:
            MultivariateGaussianEmission(std::vector<arma::vec> means,
                    std::vector<arma::mat> covs);

            double loglikelihood(int state,
                    const arma::mat &single_obs) const;

            void fitFromLabels(arma::field<arma::mat> &observations,
                    arma::ivec &labels);

        private:
            std::vector<arma::vec> means_;
            std::vector<arma::mat> covs_;
    };

};

#endif

