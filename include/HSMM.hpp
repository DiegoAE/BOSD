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

            // Inits all the pmfs with the uniform distribution.
            HSMM(std::shared_ptr<AbstractEmission> emission, int nstates,
                    int ndurations, int min_duration);

            void setDuration(arma::mat duration);

            void setEmission(std::shared_ptr<AbstractEmission> emission);

            void setPi(arma::vec pi);

            void setTransition(arma::mat transition);

            void setDurationLearningChoice(std::string choice);

            void setDurationDirichletPrior(arma::mat alphas);

            void setPiFromLabels(arma::field<arma::ivec> seqs);

            void setTransitionFromLabels(arma::field<arma::ivec> seqs);

            void setDurationFromLabels(arma::field<arma::ivec> seqs);

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

            double loglikelihood(const arma::field<arma::field<
                    arma::mat>>& obs);

            double loglikelihood(
                    const arma::field<arma::field<arma::mat>>& obs,
                    const arma::field<Labels> &observed_segments);

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

            // Parameters of the Dirichlet prior assumed over the duration.
            // By default no prior is assumed \alpha = 1.
            arma::mat dirichlet_alphas_;

            std::shared_ptr<AbstractEmission> emission_;
            std::string duration_learning_choice_;
            bool learning_transitions_;
            bool learning_pi_;
            bool debug_;

        protected:

            std::pair<arma::ivec, arma::ivec>
                    computeViterbiStateDurationSequenceFromLabels(
                    arma::ivec seq);

            // Approximates a discrete (truncate) Gaussian to a given duration.
            arma::mat gaussianMomentMatching(arma::mat duration) const;

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

            void sampleFromPosterior(int &dur, int &offset, int &hs) const;

            arma::field<arma::mat> sampleNextObservations(int nobs) const;

            void printTopKFromPosterior(int k) const;

            arma::vec getStateMarginal() const;

            arma::vec getRunlengthMarginal() const;

            arma::vec getDurationMarginal() const;

            arma::vec getImplicitDurationMarginal() const;

        protected:

            arma::mat getDurationSuffixSum() const;

            int appendToField(arma::field<arma::mat>& current_obs, int idx,
                    const arma::field<arma::mat> new_obs) const;

            // Logposterior over (d (duration),s (offset),i (hidden state))
            // conditioned on all the observations given so far.
            arma::cube last_log_posterior_;
            std::vector<arma::mat> observations_;
            std::deque<arma::vec> alpha_posteriors_;
    };


    class OnlineHSMMRunlengthBased : public HSMM {
        public:
            OnlineHSMMRunlengthBased(std::shared_ptr<
                    AbstractEmissionObsCondIIDgivenState> emission,
                    arma::mat transition, arma::vec pi, arma::mat duration,
                    int min_duration);

            OnlineHSMMRunlengthBased(std::shared_ptr<
                    AbstractEmissionObsCondIIDgivenState> emission,
                    int nstates, int ndurations, int min_duration);

            void init();

            std::shared_ptr<AbstractEmissionObsCondIIDgivenState>
                    getOnlineDurationAgnosticEmission() const;

            void addNewObservation(const arma::mat& obs);

            arma::vec getRunlengthMarginal() const;

            arma::vec getStateMarginal() const;

            arma::vec getResidualTimeMarginal() const;

            // Based on last_residualtime_posterior_
            arma::vec getStateMarginal2() const;

            double oneStepAheadLoglikelihood(const arma::mat& obs) const;

        protected:

            double loglikelihood_(int state, const arma::mat& obs) const;

            arma::mat getDurationSuffixSum_() const;

            // Hazard(r): probability that the segment ends "now" given that
            // the current runlength is r.
            arma::mat getHazardFunction_() const;

            // Posterior over (r (runlength a.k.a. offset), i (hidden state))
            arma::mat last_posterior_;

            // Posterior over (l (residual time), i, (hidden state))
            arma::mat last_residualtime_posterior_;

            std::vector<arma::mat> observations_;
    };

};

#endif
