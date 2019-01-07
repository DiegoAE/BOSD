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

            MultivariateGaussianEmission* clone() const;

            double loglikelihood(int state,
                    const arma::vec &single_obs) const;

            void fitFromLabels(const arma::field<arma::mat> &observations_seq,
                    const arma::field<arma::ivec> &labels_seq);

            // TODO: throw a not implemented exception.
            void reestimate(int min_duration,
                    const arma::field<arma::cube>& meta,
                    const arma::field<arma::field<arma::mat>>& mobs) {}

            arma::field<arma::mat> sampleFromState(int state, int size,
                    std::mt19937 &rng) const;

        private:
            std::vector<robotics::random::NormalDist> states_;
    };

};

#endif

