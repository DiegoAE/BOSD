#ifndef HSMM_H
#define HSMM_H

#include <armadillo>
#include <cmath>
#include <iostream>
#include <json.hpp>
#include <memory>
#include <set>
#include <vector>

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
            virtual void reestimate(int min_duration, const arma::cube& eta,
                    const arma::mat& obs) = 0;

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

            void reestimate(int min_duration, const arma::cube& eta,
                    const arma::mat& obs);

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

            void reestimate(int min_duration, const arma::cube& eta,
                    const arma::mat& obs);

            arma::mat sampleFromState(int state, int size) const;

        private:
            double std_dev_output_noise_;
            arma::mat means_;
    };


    class ObservedSegment {
        public:
            ObservedSegment(int t, int d);

            ObservedSegment(int t, int d, int hidden_state);

            int getDuration() const;

            int getEndingTime() const;

            int getHiddenState() const;

            int getStartingTime() const;

            bool operator< (const ObservedSegment & segment) const;

        private:
            int t_;
            int d_;
            int hidden_state_;
    };


    class Labels {
        public:
            Labels();

            // Sets a segment observation ending at t with duration d.
            void setLabel(int t, int d);

            // As above but additionaly specifies the generating hidden state.
            void setLabel(int t, int d, int hidden_state);

        private:
            std::set<ObservedSegment> labels_;
    };


    class HSMM {
        public:
            HSMM(std::shared_ptr<AbstractEmission> emission,
                    arma::mat transition, arma::vec pi, arma::mat duration,
                    int min_duration);

            void setDuration(arma::mat duration);

            void setEmission(std::shared_ptr<AbstractEmission> emission);

            void setPi(arma::vec pi);

            void setTransition(arma::mat transition);

            arma::mat sampleSegments(int nsegments, arma::ivec& hiddenStates,
                    arma::ivec& hiddenDurations);

            // Fits the model w.r.t. obs. Returns true if it reaches
            // convergence.
            bool fit(arma::mat obs, int max_iter, double tol);

            // Computes the likelihoods w.r.t. the emission model.
            arma::cube computeEmissionsLikelihood(const arma::mat obs);

            // Computes the loglikelihoods w.r.t. the emission model.
            arma::cube computeEmissionsLogLikelihood(const arma::mat obs);

            // Returns a json representation of the model.
            nlohmann::json to_stream() const;

            // Reads the parameters from a json file.
            void from_stream(const nlohmann::json &params);

            arma::mat transition_;
            arma::vec pi_;
            arma::mat duration_;
            int ndurations_;
            int min_duration_;
            int nstates_;
            std::shared_ptr<AbstractEmission> emission_;
    };

};

#endif
