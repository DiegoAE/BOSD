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

    class NNEmission : public AbstractEmissionConditionalIIDobs {
        public:
            NNEmission(int nstates, int njoints,
                    arma::ivec hidden_units_per_layer);

            NNEmission* clone() const;

            double loglikelihoodIIDobs(int state, int seg_dur, int offset,
                    const arma::mat& single_obs) const;

            void reestimate(int min_duration,
                    const arma::field<arma::cube>& meta,
                    const arma::field<arma::field<arma::mat>>& mobs);

            arma::field<arma::mat> sampleFromState(int state, int size,
                    std::mt19937 &rng) const;

            arma::field<arma::mat> sampleNextObsGivenPastObs(int state,
                    int seg_dur, const arma::field<arma::mat>& past_obs,
                    std::mt19937 &rng) const;

            NNmodel& getNeuralNet(int state);

            void setNoiseVar(const arma::mat& noise_var);

        protected:

            arma::vec getSampleLocations(int length) const;

            mutable std::vector<NNmodel> ffns_;
            arma::mat noise_var_;

            // Delta for emissions which are not dependent on the total
            // duration.
            double sample_locations_delta_ = -1;
    };

};
#endif
