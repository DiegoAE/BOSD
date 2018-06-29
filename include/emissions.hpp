#ifndef EMISSIONS_H
#define EMISSIONS_H

#include <armadillo>
#include <json.hpp>

namespace hsmm {

    class AbstractEmission {
        public:
            AbstractEmission(int nstates, int dimension);

            int getNumberStates() const;

            int getDimension() const;

            virtual AbstractEmission* clone() const = 0;

            virtual double loglikelihood(int state, const arma::mat& obs)
                    const = 0;

            arma::cube likelihoodCube(int min_duration, int ndurations,
                    const arma::mat &obs) const;

            // This should return a cube of dimensions (nstates, nobs,
            // ndurations) where the entry (i, j, k) is the log-likelihood of
            // the observations in the interval [j, min_duration + k - 1]
            // being produced by state i.
            virtual arma::cube loglikelihoodCube(int min_duration,
                    int ndurations, const arma::mat& obs) const;

            virtual nlohmann::json to_stream() const;

            virtual void from_stream(const nlohmann::json &emission_params);

            // Reestimates in place the emission parameters using the
            // statistics provided by the HSMM E step. eta(j, d, t) represents
            // the expected value of state j generating a segment of length
            // min_duration + d ending at time t.
            virtual void reestimate(int min_duration,
                    const arma::field<arma::cube>& meta,
                    const arma::field<arma::mat>& mobs) = 0;

            virtual arma::mat sampleFromState(int state, int size) const = 0;

        private:
            int nstates_;
            int dimension_;
    };


    class DummyGaussianEmission : public AbstractEmission {
        public:
            DummyGaussianEmission(arma::vec& means, arma::vec& std_devs);

            DummyGaussianEmission* clone() const;

            double loglikelihood(int state, const arma::mat& obs) const;

            virtual nlohmann::json to_stream() const;

            void reestimate(int min_duration,
                    const arma::field<arma::cube>& meta,
                    const arma::field<arma::mat>& mobs);

            arma::mat sampleFromState(int state, int size) const;

        private:
            arma::vec means_;
            arma::vec std_devs_;
    };


    class DummyMultivariateGaussianEmission : public AbstractEmission {
        public:
            DummyMultivariateGaussianEmission(arma::mat& means,
                    double std_dev_output_noise);

            DummyMultivariateGaussianEmission* clone() const;

            double loglikelihood(int state, const arma::mat& obs) const;

            void reestimate(int min_duration,
                    const arma::field<arma::cube>& meta,
                    const arma::field<arma::mat>& mobs);

            arma::mat sampleFromState(int state, int size) const;

        private:
            double std_dev_output_noise_;
            arma::mat means_;
    };

};

#endif
