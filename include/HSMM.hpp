#ifndef HSMM_H
#define HSMM_H

#include <armadillo>
#include <deque>
#include <emissions.hpp>
#include <ForwardBackward.hpp>
#include <json.hpp>
#include <memory>
#include <vector>

namespace hsmm {

    class HSMM {
        public:
            HSMM(std::shared_ptr<AbstractEmission> emission,
                    arma::mat transition, arma::vec pi, arma::mat duration,
                    int min_duration);

            void setDuration(arma::mat duration);

            void setEmission(std::shared_ptr<AbstractEmission> emission);

            void setPi(arma::vec pi);

            void setTransition(arma::mat transition);

            arma::field<arma::mat> sampleSegments(int nsegments,
                    arma::ivec& hiddenStates,
                    arma::ivec& hiddenDurations);

            arma::field<arma::field<arma::mat>> sampleMultipleSequences(
                    int nsequences, int nsegments,
                    arma::field<arma::ivec>& seqsHiddenStates,
                    arma::field<arma::ivec>& seqsHiddenDurations);

            void init_params_from_data(const arma::field<arma::field<
                    arma::mat>> &obs);

            // Fits the model from multiple sequences.
            bool fit(arma::field<arma::field<arma::mat>> obs,
                    int max_iter, double tol);

            // As above (multiple sequences) but also accepts labels for
            // semi-supervised learning.
            bool fit(arma::field<arma::field<arma::mat>> obs,
                    arma::field<Labels> observed_segments, int max_iter,
                    double tol);

            // Computes the likelihoods w.r.t. the emission model.
            arma::cube computeEmissionsLikelihood(
                    const arma::field<arma::mat>& obs);

            // Computes the loglikelihoods w.r.t. the emission model.
            arma::cube computeEmissionsLogLikelihood(
                    const arma::field<arma::mat>& obs);

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
            bool learn_duration_;
            bool debug_;

        protected:

            double lower_bound_term_transition(const arma::field<
                    arma::cube> &zetas, const arma::mat& log_transition) const;

            double lower_bound_term_pi(const arma::field<arma::cube> &etas,
                    const arma::vec& log_pi) const;

            double lower_bound_term_duration(const arma::field<
                    arma::cube> &etas, const arma::mat& log_duration) const;
    };


    class OnlineHSMM : public HSMM {
        public:
            OnlineHSMM(std::shared_ptr<AbstractEmissionOnlineSetting> emission,
                    arma::mat transition, arma::vec pi, arma::mat duration,
                    int min_duration);

            std::shared_ptr<AbstractEmissionOnlineSetting> getOnlineEmission(
                    ) const;

            void addNewObservation(const arma::mat& obs);

            arma::mat sampleNextObservation() const;

        protected:

            // Logposterior over (d (duration),s (offset),i (hidden state))
            // conditioned on all the observations given so far.
            arma::cube last_log_posterior_;
            std::vector<arma::mat> observations_;
            std::deque<arma::vec> alpha_posteriors_;
    };

};

#endif
