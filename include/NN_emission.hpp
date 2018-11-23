/**
 * NOTE: This emission process is not being used for now. The NN basis function
 * is used instead.
 */

#ifndef NN_EMISSION_H
#define NN_EMISSION_H

#include <mlpack/core.hpp>

// The following include will be redundant in the newer versions of mlpack.
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <armadillo>
#include <emissions.hpp>


namespace hsmm {

    typedef mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>,
            mlpack::ann::RandomInitialization> NNmodel;

    class NNEmission : public AbstractEmissionOnlineSetting {
        public:
            NNEmission(int nstates, int njoints);

            NNEmission* clone() const;

            double loglikelihood(int state,
                    const arma::field<arma::mat>& obs) const;

            void reestimate(int min_duration,
                    const arma::field<arma::cube>& meta,
                    const arma::field<arma::field<arma::mat>>& mobs);

            arma::field<arma::mat> sampleFromState(int state, int size,
                    std::mt19937 &rng) const;

            arma::field<arma::mat> sampleNextObsGivenPastObs(int state,
                    int seg_dur, const arma::field<arma::mat>& past_obs,
                    std::mt19937 &rng) const;

        protected:

            arma::vec getSampleLocations(int length) const;

            std::vector<NNmodel> ffns_;
            arma::mat noise_var_;

            // Delta for emissions which are not dependent on the total
            // duration.
            double sample_locations_delta_ = -1;
    };

};
#endif
