#ifndef HSMM_H
#define HSMM_H

#include <armadillo>
#include <cassert>
#include <cmath>
#include <iostream>
#include <json.hpp>
#include <memory>
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

            void fit(arma::mat obs, int max_iter, double tol);

            // Computes the likelihoods w.r.t. the emission model.
            arma::cube computeEmissionsLikelihood(const arma::mat obs);

            // Computes the loglikelihoods w.r.t. the emission model.
            arma::cube computeEmissionsLogLikelihood(const arma::mat obs);

            arma::mat transition_;
            arma::vec pi_;
            arma::mat duration_;
            int ndurations_;
            int min_duration_;
            int nstates_;
            std::shared_ptr<AbstractEmission> emission_;
    };

    // TODO: incorporate this into the HSMM.
    void viterbiPath(const arma::imat& psi_d, const arma::imat& psi_s,
            const arma::mat& delta, arma::ivec& hiddenStates,
            arma::ivec& hiddenDurations);

};

#endif
