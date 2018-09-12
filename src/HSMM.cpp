#include <armadillo>
#include <ForwardBackward.hpp>
#include <HSMM.hpp>
#include <cmath>
#include <json.hpp>
#include <iostream>
#include <memory>
#include <vector>

using namespace arma;
using namespace std;
using json = nlohmann::json;

namespace hsmm {

    // TODO: Make sure there is not any bias here.
    int sampleFromCategorical(rowvec pmf) {
        rowvec prefixsum(pmf);
        for(int i = 1; i < pmf.n_elem; i++)
            prefixsum(i) += prefixsum(i - 1);
        assert(abs(prefixsum(pmf.n_elem - 1) - 1.0) < 1e-7);
        return lower_bound(prefixsum.begin(), prefixsum.end(), randu()) -
                prefixsum.begin();
    }

    /**
     * HSMM implementation.
     */
    HSMM::HSMM(shared_ptr<AbstractEmission> emission, mat transition,
            vec pi, mat duration, int min_duration) : emission_(emission),
            learn_duration_(true), debug_(false) {
        nstates_ = emission_->getNumberStates();
        ndurations_ = duration.n_cols;
        min_duration_ = min_duration;
        assert(min_duration_ >= 1);
        assert(nstates_ >= 1);
        assert(ndurations_ >= 1);
        setDuration(duration);
        setPi(pi);
        setTransition(transition);
    }

    void HSMM::setDuration(mat duration) {
        assert(duration.n_rows == nstates_);
        assert(duration.n_cols == ndurations_);
        duration_ = duration;
    }

    void HSMM::setEmission(shared_ptr<AbstractEmission> emission) {
        assert(emission->getNumberStates() == nstates_);
        emission_ = emission;
    }

    void HSMM::setPi(vec pi) {
        assert(pi.n_elem == nstates_);
        pi_ = pi;
    }

    void HSMM::setTransition(mat transition) {
        assert(transition.n_rows == transition.n_cols);
        assert(transition.n_rows == nstates_);
        transition_ = transition;
    }

    field<mat> HSMM::sampleSegments(int nsegments, ivec& hiddenStates,
            ivec& hiddenDurations) {
        assert(nsegments >= 1);

        // Generating states sequence.
        ivec states(nsegments);
        states(0) = sampleFromCategorical(pi_.t());
        for(int i = 1; i < nsegments; i++) {
            rowvec nstateDist = transition_.row(states(i - 1));
            states(i) = sampleFromCategorical(nstateDist);
        }

        // Generating durations from states.
        int sampleSequenceLength = 0;
        ivec durations(nsegments);
        for(int i = 0; i < nsegments; i++) {
            int currentDuration = sampleFromCategorical(
                    duration_.row(states(i))) + min_duration_;
            durations(i) = currentDuration;
            sampleSequenceLength += currentDuration;
        }

        // Generating samples
        field<mat> samples(sampleSequenceLength);
        int idx = 0;
        for(int i = 0; i < nsegments; i++) {
            field<mat> currSample = emission_->sampleFromState(
                    states(i), durations(i));
            samples.rows(idx, idx + durations(i) - 1) = currSample;
            idx += durations(i);
        }
        hiddenStates = states;
        hiddenDurations = durations;
        return samples;
    }

    field<field<mat>> HSMM::sampleMultipleSequences(int nsequences,
            int nsegments, field<ivec>& seqsHiddenStates,
            field<ivec>& seqsHiddenDurations) {
        field<field<mat>> mobs(nsequences);
        field<ivec> seqsHS(nsequences);
        field<ivec> seqsDur(nsequences);
        for(int s = 0; s < nsequences; s++) {
            ivec hs, dur;
            mobs(s) = sampleSegments(nsegments, hs, dur);
            seqsHS(s) = hs;
            seqsDur(s) = dur;
        }
        seqsHiddenStates = seqsHS;
        seqsHiddenDurations = seqsDur;
        return mobs;
    }

    void HSMM::init_params_from_data(const field<field<mat>> &mobs) {
        emission_->init_params_from_data(min_duration_, ndurations_, mobs);
    }

    bool HSMM::fit(field<field<mat>> mobs, field<Labels> mobserved_segments,
            int max_iter, double tol) {

        // Array initializations.
        int nseq = mobs.n_elem;
        assert(nseq >= 1 && nseq == mobserved_segments.n_elem);
        field<mat> malpha(nseq);
        field<mat> mbeta(nseq);
        field<mat> malpha_s(nseq);
        field<mat> mbeta_s(nseq);
        field<vec> mbeta_s_0(nseq);
        field<cube> meta(nseq);
        field<cube> mzeta(nseq);
        for(int i = 0; i < nseq; i++) {
            int nobs = mobs(i).n_elem;
            assert(nobs >= min_duration_);
            malpha(i) = zeros<mat>(nstates_, nobs);
            mbeta(i) = zeros<mat>(nstates_, nobs);
            malpha_s(i) = zeros<mat>(nstates_, nobs);
            mbeta_s(i) = zeros<mat>(nstates_, nobs);
            mbeta_s_0(i) = zeros<vec>(nstates_);
            meta(i) = zeros<cube>(nstates_, ndurations_, nobs);
            mzeta(i) = zeros<cube>(nstates_, nstates_, nobs - 1);
        }

        mat log_estimated_transition = log(transition_);
        vec log_estimated_pi = log(pi_);
        mat log_estimated_duration = log(duration_);
        double marginal_llikelihood = -datum::inf;
        bool convergence_reached = false;
        for(int i = 0; i < max_iter && !convergence_reached; i++) {
            for(int s = 0; s < nseq; s++) {
                const field<mat>& obs = mobs(s);
                const Labels& observed_segments = mobserved_segments(s);

                // Assertions if labels are provided.
                if (!observed_segments.empty()) {
                    for(const auto &segment : observed_segments.getLabels())
                        assert(segment.getDuration() >= min_duration_ &&
                                segment.getDuration() < min_duration_ +
                                ndurations_);
                    int start_time = observed_segments.getFirstSegment(
                            ).getStartingTime();
                    int end_time = observed_segments.getLastSegment(
                            ).getEndingTime();
                    if (start_time > 0)
                        assert(start_time >= min_duration_);
                    if (end_time < obs.n_elem - 1)
                        assert(obs.n_elem - end_time > min_duration_);
                }

                mat& alpha = malpha(s);
                mat& beta = mbeta(s);
                mat& alpha_s = malpha_s(s);
                mat& beta_s = mbeta_s(s);
                vec& beta_s_0 = mbeta_s_0(s);
                cube& eta = meta(s);
                cube& zeta = mzeta(s);

                // Recomputing the emission likelihoods.
                cube logpdf = computeEmissionsLogLikelihood(obs);

                logsFB(log_estimated_transition, log_estimated_pi,
                        log_estimated_duration, logpdf, observed_segments,
                        alpha, beta, alpha_s, beta_s, beta_s_0, eta, zeta,
                        min_duration_, obs.n_elem);
            }
            vec sequences_llikelihood(nseq);
            for(int s = 0; s < nseq; s++) {
                int nobs = mobs(s).n_elem;

                // Computing the marginal likelihood (aka observation
                // likelihood).
                sequences_llikelihood(s) = logsumexp(malpha(s).col(nobs - 1));
            }
            double current_llikelihood = sum(sequences_llikelihood);
            cout << "EM iteration " << i << " marginal log-likelihood: " <<
                    current_llikelihood << ". Diff: " <<
                    current_llikelihood - marginal_llikelihood << endl;
            if (current_llikelihood < marginal_llikelihood)
                cout << "Warning: The log-likelihood decreased probably due" <<
                        " to numerical errors." << endl;
            if (current_llikelihood - marginal_llikelihood < tol) {
                convergence_reached = true;
                marginal_llikelihood = current_llikelihood;
                break;
            }
            marginal_llikelihood = current_llikelihood;

            // M step.
            // Lower bound before M step.
            double lb_pi, lb_transition, lb_duration;
            if (debug_) {
                lb_pi = lower_bound_term_pi(meta, log_estimated_pi);
                lb_transition = lower_bound_term_transition(mzeta,
                        log_estimated_transition);
                lb_duration = lower_bound_term_duration(meta,
                        log_estimated_duration);
            }

            mat tmp_transition(log_estimated_transition);
            for(int i = 0; i < nstates_; i++) {
                vector<double> den;
                for(int j = 0; j < nstates_; j++) {
                    vector<double> num;
                    for(int s = 0; s < nseq; s++) {
                        const cube& zeta = mzeta(s);
                        for(int t = 0; t < mobs(s).n_elem - 1; t++) {
                            num.push_back(zeta(i, j, t));
                            den.push_back(zeta(i, j, t));
                        }
                    }
                    vec num_v(num);
                    if (num_v.n_elem > 0)
                        tmp_transition(i, j) = logsumexp(num_v);
                }
                vec den_v(den);
                if (den_v.n_elem > 0) {
                    double denominator = logsumexp(den_v);

                    // Handling the case when the transition probability is 0.
                    if (denominator != -datum::inf) {
                        for(int j = 0; j < nstates_; j++)
                            tmp_transition(i, j) -= denominator;
                    }
                }
            }
            log_estimated_transition = tmp_transition;

            // Reestimating the initial state pmf.
            vec tmp_pi(size(pi_), fill::zeros);
            for(const vec& beta_s_0 : mbeta_s_0) {
                // TODO: fix the following line to take into account the labels.
                vec current_log_estimated_pi = beta_s_0 + log_estimated_pi;
                // double mllh = log(sum(estimated_pi));
                current_log_estimated_pi = current_log_estimated_pi -
                        logsumexp(current_log_estimated_pi);
                vec current_pi = exp(current_log_estimated_pi);
                assert(abs(sum(current_pi) - 1) < 1e-7);
                tmp_pi += current_pi;
            }
            log_estimated_pi = log(tmp_pi / nseq);

            // Reestimating durations.
            // D(j, d) represents the expected number of times that state
            // j is visited with duration d (non-normalized).
            mat D(size(duration_), fill::zeros);
            for(int i = 0; i < nstates_; i++) {
                vector<double> den;
                for(int d = 0; d < ndurations_; d++) {
                    vector<double> ts;
                    for(int s = 0; s < nseq; s++) {
                        int nobs = mobs(s).n_rows;
                        const cube& eta = meta(s);
                        for(int t = 0; t < nobs; t++) {
                            ts.push_back(eta(i, d, t));
                            den.push_back(eta(i, d, t));
                        }
                    }
                    vec ts_v(ts);
                    D(i, d) = logsumexp(ts_v);
                }
                vec den_v(den);
                double denominator = logsumexp(den_v);

                // Handling the case when the transition probability mass is 0.
                if (denominator != -datum::inf) {
                    for(int d = 0; d < ndurations_; d++)
                        D(i, d) -= denominator;
                }
            }
            if (learn_duration_)
                log_estimated_duration = D;

            // Lower bound after M step.
            if (debug_) {
                double lb_pi_after = lower_bound_term_pi(meta, log_estimated_pi);
                double lb_transition_after = lower_bound_term_transition(mzeta,
                        log_estimated_transition);
                double lb_duration_after = lower_bound_term_duration(meta,
                        log_estimated_duration);
                cout << "Diff lower bound before and after M step" << endl;
                cout << "pi: " << lb_pi_after - lb_pi << endl;
                cout << "transition: " << lb_transition_after - lb_transition << endl;
                cout << "duration: " << lb_duration_after - lb_duration << endl;
                assert(lb_pi_after - lb_pi > -tol);
                assert(lb_transition_after - lb_transition > -tol);
                assert(lb_duration_after - lb_duration > -tol);
            }

            // Reestimating emissions.
            // NOTE: the rest of the HSMM parameters are updated out of
            // this loop.
            emission_->reestimate(min_duration_, meta, mobs);
        }

        cout << "Stopped because of " << ((convergence_reached) ?
                "convergence." : "max iter.") << endl;

       // Updating the model parameters.
       setTransition(exp(log_estimated_transition));
       setPi(exp(log_estimated_pi));
       setDuration(exp(log_estimated_duration));
       return convergence_reached;
    }

    bool HSMM::fit(field<field<mat>> mobs, int max_iter, double tol) {
        field<Labels> mobserved_segments(mobs.n_elem);  // empty.
        return fit(mobs, mobserved_segments, max_iter, tol);
    }

    // Computes the likelihoods w.r.t. the emission model.
    cube HSMM::computeEmissionsLikelihood(const field<mat>& obs) {
        return emission_->likelihoodCube(min_duration_, ndurations_, obs);
    }

    // Computes the loglikelihoods w.r.t. the emission model.
    cube HSMM::computeEmissionsLogLikelihood(const field<mat>& obs) {
        return emission_->loglikelihoodCube(min_duration_, ndurations_,
            obs);
    }

    // Returns a json representation of the model.
    nlohmann::json HSMM::to_stream() const {
        nlohmann::json ret;
        ret["nstates"] = nstates_;
        ret["min_duration"] = min_duration_;
        ret["ndurations"] = ndurations_;
        ret["initial_pmf"] = pi_;
        ret["emission_params"] = emission_->to_stream();

        // Taking care of the serialization of armadillo matrices.
        vector<vector<double>> transition_v, duration_v;
        for(int i = 0; i < nstates_; i++) {
            transition_v.push_back(conv_to<vector<double>>::from(
                    transition_.row(i)));
            duration_v.push_back(conv_to<vector<double>>::from(
                    duration_.row(i)));
        }
        ret["transition"] = transition_v;
        ret["duration"] = duration_v;
        return ret;
    }

    // Reads the parameters from a json file.
    void HSMM::from_stream(const nlohmann::json &params) {
        nstates_ = params.at("nstates");
        min_duration_ = params.at("min_duration");
        ndurations_ = params.at("ndurations");
        const nlohmann::json& emission_params = params.at("emission_params");
        emission_->from_stream(emission_params);

        // Parsing the armadillo matrices (transition, pi, duration).
        vector<double> initial_pmf_v = params.at("initial_pmf");
        vector<vector<double>> transition_v = params.at("transition");
        vector<vector<double>> duration_v = params.at("duration");
        vec pi = conv_to<vec>::from(initial_pmf_v);
        mat transition = zeros<mat>(nstates_, nstates_);
        mat duration = zeros<mat>(nstates_, ndurations_);
        for(int i = 0; i < nstates_; i++) {
            transition.row(i) = conv_to<rowvec>::from(transition_v.at(i));
            duration.row(i) = conv_to<rowvec>::from(duration_v.at(i));
        }
        setPi(pi);
        setTransition(transition);
        setDuration(duration);
    }

    // Debugging functions.
    double HSMM::lower_bound_term_transition(const field<cube>& zetas,
            const mat& log_transition) const {
        double ret = 0;
        for(const cube& zeta : zetas)
            for(int i = 0; i < nstates_; i++)
                for(int j = 0; j < nstates_; j++)
                    for(int t = 0; t < zeta.n_slices; t++)
                        if (log_transition(i, j) > -datum::inf)
                            ret += exp(zeta(i, j, t)) * log_transition(i, j);
       return ret;
    }

    double HSMM::lower_bound_term_pi(const field<cube>& etas,
            const vec& log_pi) const {
        double ret = 0;
        for(const cube& eta : etas)
            for(int i = 0; i < nstates_; i++)
                for(int d = 0; d < ndurations_; d++)
                    if (log_pi(i) > -datum::inf)
                        ret += exp(eta(i, d, min_duration_ + d - 1)) *
                            log_pi(i);
        return ret;
    }

    double HSMM::lower_bound_term_duration(const field<cube>& etas,
            const mat& log_duration) const {
        double ret = 0;
        for(const cube& eta : etas)
            for(int i = 0; i < nstates_; i++)
                for(int d = 0; d < ndurations_; d++)
                    for(int t = 0; t < eta.n_slices; t++)
                        if (log_duration(i, d) > -datum::inf)
                            ret += exp(eta(i, d, t)) * log_duration(i, d);
        return ret;
    }


    /*
     * Online HSMM implementation
     */
    OnlineHSMM::OnlineHSMM(shared_ptr<AbstractEmissionOnlineSetting> emission,
            mat transition, vec pi, mat duration, int min_duration) : HSMM(
            static_pointer_cast<AbstractEmission>(emission), transition,
            pi, duration, min_duration),
            last_log_posterior_(ndurations_, min_duration_ + ndurations_,
            nstates_) {
        alpha_posteriors_.push_back(log(pi_));
    }

    void OnlineHSMM::addNewObservation(const mat& obs) {
        observations_.push_back(obs);
        mat log_duration = log(duration_);
        mat log_transition = log(transition_);
        last_log_posterior_.fill(-datum::inf);
        vector<double> normalization_terms;
        for(int d = min_duration_; d < min_duration_ + ndurations_; d++) {
            for(int s = 0; s < d; s++) {

                // Making sure the offset is consistent with the number of
                // observations so far.
                if (s >= observations_.size())
                    break;

                // Building the current segment. Notice that it is padded with
                // empty matrices.
                field<mat> current_segment(d);
                for(int idx = s, obs_idx = observations_.size()-1; idx >= 0;
                        idx--, obs_idx--) {
                    current_segment(idx) = observations_[obs_idx];
                }
                const vec& relevant_alpha_posterior = alpha_posteriors_.at(
                        alpha_posteriors_.size() - 1 - s);
                for(int i = 0; i < nstates_; i++) {
                    double log_pdf_seg = emission_->loglikelihood(i,
                            current_segment) + log_duration(i,d-min_duration_);
                    double log_unnormalized_value = log_pdf_seg +
                            relevant_alpha_posterior(i);
                    last_log_posterior_(d - min_duration_, s, i) =
                            log_unnormalized_value;
                    normalization_terms.push_back(log_unnormalized_value);
                }
            }
        }

        // Normalizing the current joint posterior.
        double normalization_c = logsumexp(normalization_terms);
        assert(normalization_c > -datum::inf);
        last_log_posterior_ -= normalization_c;

        // Computing the marginal posterior over the next hidden state assuming
        // there is a change point right after the current input observation.
        vec current_marginal_posterior(nstates_);
        for(int i = 0; i < nstates_; i++) {
            vector<double> terms;
            for(int j = 0; j < nstates_; j++) {
                for(int dur = 0; dur < ndurations_; dur++) {
                    double term = last_log_posterior_(dur,
                            min_duration_ + dur - 1, j) + log_transition(j, i);
                    terms.push_back(term);
                }
            }
            current_marginal_posterior(i) = logsumexp(terms);
        }

        // Pushing back the computed posterior.
        alpha_posteriors_.push_back(current_marginal_posterior);
    }

    mat OnlineHSMM::sampleNextObservation() const {
        cube current_posterior = exp(last_log_posterior_);
        int nrows = current_posterior.n_rows;
        int ncols = current_posterior.n_cols;
        int nslices = current_posterior.n_slices;

        // Sampling a triplet (d,s,i) from the current posterior.
        double accum = 0.0;
        double usample = randu();
        int found = 0;
        mat sample;
        for(int i = 0; i < nslices; i++) {
            for(int d = 0; d < nrows; d++) {
                for(int s = 0; s < ncols; s++) {
                    double naccum = accum + current_posterior(d, s, i);
                    if (accum < usample && usample < naccum) {
                        found++;
                        if (s == min_duration_ + d - 1) {

                            // Handling the case when there is a transition
                            // right after this. Sampling next state and
                            // duration.
                            int next_state = sampleFromCategorical(
                                    transition_.row(i));
                            int next_duration = sampleFromCategorical(
                                    duration_.row(next_state)) + min_duration_;
                            field<mat> no_past_obs;
                            sample = static_pointer_cast<
                                    AbstractEmissionOnlineSetting>(
                                    emission_)->sampleNextObsGivenPastObs(
                                    next_state, next_duration, no_past_obs);
                        }
                        else {
                            field<mat> last_obs(s + 1);
                            int tam = observations_.size();
                            for(int j = s, idx = tam - 1; j >= 0; j--, idx--)
                                last_obs(j) = observations_.at(idx);
                            sample = static_pointer_cast<
                                    AbstractEmissionOnlineSetting>(
                                    emission_)->sampleNextObsGivenPastObs(i,
                                    min_duration_ + d, last_obs);
                        }
                    }
                    accum = naccum;
                }
            }
        }
        assert(found == 1);
        assert(abs(accum - 1.0) < 1e-7);
        assert(!sample.is_empty());
        return sample;
    }

};
