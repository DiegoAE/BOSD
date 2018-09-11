#ifndef EMISSIONS_H
#define EMISSIONS_H

#include <armadillo>
#include <json.hpp>
#include <random>

namespace hsmm {

    class AbstractEmission {
        public:
            AbstractEmission(int nstates, int dimension);

            int getNumberStates() const;

            int getDimension() const;

            virtual AbstractEmission* clone() const = 0;

            // Init the parameters from the provided data.
            virtual void init_params_from_data(int min_duration,
                    int ndurations, const arma::field<arma::field<
                    arma::mat>>& mobs) {}

            // Joint loglikelihood of a single segment being generated by a
            // particular state.
            virtual double loglikelihood(int state,
                    const arma::field<arma::mat>& obs) const = 0;

            arma::cube likelihoodCube(int min_duration, int ndurations,
                    const arma::field<arma::mat>& obs) const;

            // This should return a cube of dimensions (nstates, nobs,
            // ndurations) where the entry (i, j, k) is the log-likelihood of
            // the observations in the interval [j, min_duration + k - 1]
            // being produced by state i.
            virtual arma::cube loglikelihoodCube(int min_duration,
                    int ndurations, const arma::field<arma::mat>& obs) const;

            virtual nlohmann::json to_stream() const;

            virtual void from_stream(const nlohmann::json &emission_params);

            // Reestimates in place the emission parameters using the
            // statistics provided by the HSMM E step. eta(j, d, t) represents
            // the expected value of state j generating a segment of length
            // min_duration + d ending at time t.
            virtual void reestimate(int min_duration,
                    const arma::field<arma::cube>& meta,
                    const arma::field<arma::field<arma::mat>>& mobs) = 0;

            arma::field<arma::mat> sampleFromState(int state, int nsegments);

            virtual arma::field<arma::mat> sampleFromState(int state,
                    int nsegments, std::mt19937 &rng) const = 0;
        protected:

            // Pseudo-random number generation.
            std::mt19937 rand_generator_;

        private:
            int nstates_;
            int dimension_;
    };


    // For the HSMM online setting (prediction) is required to sample
    // conditioning on some already seen observations.
    class AbstractEmissionOnlineSetting : public AbstractEmission {
        public:
            AbstractEmissionOnlineSetting(int states, int dimension) :
                    AbstractEmission(states, dimension) {}

            arma::mat sampleNextObsGivenPastObs(int state, int seg_dur,
                    const arma::field<arma::mat>& past_obs);

            virtual arma::mat sampleNextObsGivenPastObs(int state, int seg_dur,
                    const arma::field<arma::mat>& past_obs,
                    std::mt19937 &rng) const = 0;
    };


    // This emission class assumes the observations are conditionally
    // independent given the duration of the segment and its position on
    // it (offset).
    class AbstractEmissionConditionalIIDobs : public AbstractEmission {
        public:
            AbstractEmissionConditionalIIDobs(int nstates, int dimension);

            double loglikelihood(int state,
                    const arma::field<arma::mat>& obs) const;

            virtual double loglikelihoodIIDobs(int state, int seg_dur, int offset,
                    const arma::mat& single_obs) const = 0;
    };


    class DummyGaussianEmission : public AbstractEmissionOnlineSetting {
        public:
            DummyGaussianEmission(arma::vec& means, arma::vec& std_devs);

            DummyGaussianEmission* clone() const;

            double loglikelihood(int state,
                    const arma::field<arma::mat>& obs) const;

            double loglikelihoodIIDobs(int state, int seg_dur, int offset,
                    const arma::mat& obs) const;

            virtual nlohmann::json to_stream() const;

            void reestimate(int min_duration,
                    const arma::field<arma::cube>& meta,
                    const arma::field<arma::field<arma::mat>>& mobs);

            arma::field<arma::mat> sampleFromState(int state, int size,
                    std::mt19937 &rng) const;

            arma::mat sampleNextObsGivenPastObs(int state, int seg_dur,
                    const arma::field<arma::mat>& past_obs,
                    std::mt19937 &rng) const;

        private:
            arma::vec means_;
            arma::vec std_devs_;
    };


    class DummyMultivariateGaussianEmission : public AbstractEmission {
        public:
            DummyMultivariateGaussianEmission(arma::mat& means,
                    double std_dev_output_noise);

            DummyMultivariateGaussianEmission* clone() const;

            double loglikelihood(int state,
                    const arma::field<arma::mat>& obs) const;

            void reestimate(int min_duration,
                    const arma::field<arma::cube>& meta,
                    const arma::field<arma::field<arma::mat>>& mobs);

            arma::field<arma::mat> sampleFromState(int state, int size,
                    std::mt19937 &rng) const;

        private:
            double std_dev_output_noise_;
            arma::mat means_;
    };

};

#endif
