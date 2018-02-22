#include <armadillo>
#include <ForwardBackward.hpp>
#include <HSMM.hpp>
#include <cassert>
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
        return lower_bound(prefixsum.begin(), prefixsum.end(), randu()) -
                prefixsum.begin();
    }

    double gaussianlogpdf_(double x, double mu, double sigma) {
        double ret = ((x - mu)*(x - mu)) / (-2*sigma*sigma);
        ret = ret - log(sqrt(2 * M_PI) * sigma);
        return ret;
    }

    double gaussianpdf_(double x, double mu, double sigma) {
        return exp(gaussianlogpdf_(x, mu, sigma));
    }


    /**
     * Abstract emission implementation.
     */
    AbstractEmission::AbstractEmission(int nstates, int dimension) :
            nstates_(nstates), dimension_(dimension) {}

    int AbstractEmission::getNumberStates() const {
        return nstates_;
    }

    int AbstractEmission::getDimension() const {
        return dimension_;
    }

    cube AbstractEmission::likelihoodCube(int min_duration, int ndurations,
            const mat &obs) const {
        return exp(loglikelihoodCube(min_duration, ndurations, obs));
    }

    // This should return a cube of dimensions (nstates, nobs, ndurations)
    // where the entry (i, j, k) is the log-likelihood of the observations
    // in the interval [j, min_duration + k - 1] being produced by state i.
    cube AbstractEmission::loglikelihoodCube(int min_duration, int ndurations,
            const mat& obs) const {
        int nobs = obs.n_cols;
        cube pdf(getNumberStates(), nobs, ndurations);
        pdf.fill(-datum::inf);
        for(int i = 0; i < getNumberStates(); i++)
            for(int t = 0; t < nobs; t++)
                for(int d = 0; d < ndurations; d++) {
                    if (t + min_duration + d > nobs)
                        break;
                    int end_idx = t + min_duration + d - 1;
                    pdf(i, t, d) = loglikelihood(i, obs.cols(t, end_idx));
                }
        return pdf;
    }

    json AbstractEmission::to_stream() const {
        return json::object();  // empty json object by default.
    }


    /**
     * DummyGaussianEmission implementation.
     */
    DummyGaussianEmission::DummyGaussianEmission(vec& means, vec& std_devs) :
            AbstractEmission(means.n_elem, 1), means_(means),
            std_devs_(std_devs) {
        assert(means_.n_elem == std_devs_.n_elem);
    }

    DummyGaussianEmission* DummyGaussianEmission::clone() const {
        return new DummyGaussianEmission(*this);
    }

    double DummyGaussianEmission::loglikelihood(int state, const mat& obs) const {
        double ret = 0;
        for(int i = 0; i < obs.n_cols; i++)
            ret += gaussianlogpdf_(obs(0, i), means_(state),
                    std_devs_(state));
        return ret;
    }

    json DummyGaussianEmission::to_stream() const {
        vector<double> means = conv_to<vector<double>>::from(means_);
        vector<double> std_devs = conv_to<vector<double>>::from(std_devs_);
        json ret;
        ret["means"] = means;
        ret["std_devs"] = std_devs;
        return ret;
    }

    void DummyGaussianEmission::reestimate(int min_duration, const cube& eta,
            const mat& obs) {
        int nobs = obs.n_cols;
        int ndurations = eta.n_cols;
        for(int i = 0; i < getNumberStates(); i++) {

            // Reestimating the mean.
            vector<double> num_mult;
            vector<double> num_obs;
            for(int t = min_duration - 1; t < nobs; t++) {
                for(int d = 0; d < ndurations; d++) {
                    int first_idx_seg = t - min_duration - d + 1;
                    if (first_idx_seg < 0)
                        break;

                    // Since the observations factorize given t, d and i.
                    for(int s = first_idx_seg; s <= t; s++) {
                        num_mult.push_back(eta(i, d, t));
                        num_obs.push_back(obs(0, s));
                    }
                }
            }
            vec num_mult_v(num_mult);
            vec num_obs_v(num_obs);
            num_mult_v = num_mult_v - logsumexp(num_mult_v);
            num_mult_v = exp(num_mult_v);
            double new_mean = dot(num_mult_v, num_obs_v);

            // Reestimating the variance.
            vector<double> num_obs_var;
            for(int t = min_duration - 1; t < nobs; t++) {
                for(int d = 0; d < ndurations; d++) {
                    int first_idx_seg = t - min_duration - d + 1;
                    if (first_idx_seg < 0)
                        break;

                    // Since the observations factorize given t, d and i.
                    for(int s = first_idx_seg; s <= t; s++) {
                        double diff = (obs(0, s) - new_mean);
                        num_obs_var.push_back(diff * diff);
                    }
                }
            }
            vec num_obs_var_v(num_obs_var);
            double new_variance = dot(num_mult_v, num_obs_var_v);

            means_(i) = new_mean;
            std_devs_(i) = sqrt(new_variance);
        }
    }

    mat DummyGaussianEmission::sampleFromState(int state, int size) const {
        return randn<mat>(1, size) * std_devs_(state) + means_(state);
    }


    /**
     * DummyMultivariateGaussianEmission implementation.
     */
    DummyMultivariateGaussianEmission::DummyMultivariateGaussianEmission(
            mat& means, double std_dev_output_noise) :
            AbstractEmission(means.n_rows, means.n_cols), means_(means),
            std_dev_output_noise_(std_dev_output_noise) {}

    DummyMultivariateGaussianEmission* DummyMultivariateGaussianEmission::
            clone() const {
        return new DummyMultivariateGaussianEmission(*this);
    }

    double DummyMultivariateGaussianEmission::loglikelihood(int state,
            const mat& obs) const {
        assert(obs.n_rows == getDimension());
        int size = obs.n_cols;
        mat copy_obs(obs);
        for(int i = 0; i < getDimension(); i++)
            copy_obs.row(i) -= linspace<rowvec>(0.0, 1.0, size) +
                    means_(state, i);
        double ret = 0.0;
        for(int i = 0; i < getDimension(); i++)
            for(int j = 0; j < size; j++)
                ret += gaussianlogpdf_(copy_obs(i, j), 0,
                        std_dev_output_noise_);
        return ret;
    }

    void DummyMultivariateGaussianEmission::reestimate(int min_duration,
            const cube& eta, const mat& obs) {
        // TODO.
    }

    mat DummyMultivariateGaussianEmission::sampleFromState(
            int state, int size) const {
        mat ret = randn<mat>(getDimension(), size) * std_dev_output_noise_;
        for(int i = 0; i < getDimension(); i++)
            ret.row(i) += linspace<rowvec>(0.0, 1.0, size) + means_(state, i);
        return ret;
    }


    /**
     * HSMM implementation.
     */
    HSMM::HSMM(shared_ptr<AbstractEmission> emission, mat transition,
            vec pi, mat duration, int min_duration) : emission_(emission) {
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

    mat HSMM::sampleSegments(int nsegments, ivec& hiddenStates,
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
        mat samples(emission_->getDimension(), sampleSequenceLength);
        int idx = 0;
        for(int i = 0; i < nsegments; i++) {
            mat currSample = emission_->sampleFromState(states(i), durations(i));
            samples.cols(idx, idx + durations(i) - 1) = currSample;
            idx += durations(i);
        }
        hiddenStates = states;
        hiddenDurations = durations;
        return samples;
    }

    void HSMM::fit(mat obs, int max_iter, double tol) {
        // For now only the transition matrix is being learned.
        int nobs = obs.n_cols;
        assert(nobs >= 1);
        mat alpha(nstates_, nobs, fill::zeros);
        mat beta(nstates_, nobs, fill::zeros);
        mat alpha_s(nstates_, nobs, fill::zeros);
        mat beta_s(nstates_, nobs, fill::zeros);
        vec beta_s_0(nstates_, fill::zeros);
        cube eta(nstates_, ndurations_, nobs, fill::zeros);
        mat log_estimated_transition = log(transition_);
        vec log_estimated_pi = log(pi_);
        mat log_estimated_duration = log(duration_);
        double marginal_llikelihood = -datum::inf;
        bool convergence_reached = false;
        for(int i = 0; i < max_iter && !convergence_reached; i++) {

            // Recomputing the emission likelihoods.
            cube logpdf = computeEmissionsLogLikelihood(obs);

            logsFB(log_estimated_transition, log_estimated_pi,
                    log_estimated_duration, logpdf, alpha, beta, alpha_s,
                    beta_s, beta_s_0, eta, min_duration_, nobs);

            // Computing the marginal likelihood (aka observation likelihood).
            double current_llikelihood = logsumexp(alpha.col(nobs - 1));

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

            // Reestimating transitions.
            mat tmp_transition(size(transition_), fill::zeros);
            for(int i = 0; i < nstates_; i++) {
                vector<double> den;
                for(int j = 0; j < nstates_; j++) {
                    vec num(nobs - 1);
                    for(int t = 0; t < nobs - 1; t++) {
                        double tmp_entry = alpha(i, t) +
                                log_estimated_transition(i, j) +
                                beta_s(j, t);
                        num(t) = tmp_entry;
                        den.push_back(tmp_entry);
                    }
                    tmp_transition(i, j) = logsumexp(num);
                }
                vec den_v(den);
                double denominator = logsumexp(den_v);
                for(int j = 0; j < nstates_; j++)
                    tmp_transition(i, j) -= denominator;
            }
            log_estimated_transition = tmp_transition;
            // Reestimating the initial state pmf.
            log_estimated_pi = beta_s_0 + log_estimated_pi;
            // double mllh = log(sum(estimated_pi));
            log_estimated_pi = log_estimated_pi -
                    logsumexp(log_estimated_pi);

            // Reestimating durations.
            // D(j, d) represents the expected number of times that state
            // j is visited with duration d (non-normalized).
            mat D(size(duration_), fill::zeros);
            for(int i = 0; i < nstates_; i++) {
                vector<double> den;
                for(int d = 0; d < ndurations_; d++) {
                    vec ts(nobs);
                    for(int t = 0; t < nobs; t++) {
                        ts(t) = eta(i, d, t);
                        den.push_back(eta(i, d, t));
                    }
                    D(i, d) = logsumexp(ts);
                }
                vec den_v(den);
                double denominator = logsumexp(den_v);
                for(int d = 0; d < ndurations_; d++)
                    D(i, d) -= denominator;
            }
            log_estimated_duration = D;

            // Reestimating emissions.
            // NOTE: the rest of the HSMM parameters are updated out of
            // this loop.
            emission_->reestimate(min_duration_, eta, obs);
        }

        cout << "Stopped because of " << ((convergence_reached) ?
                "convergence." : "max iter.") << endl;

        // Updating the model parameters.
       setTransition(exp(log_estimated_transition));
       setPi(exp(log_estimated_pi));
       setDuration(exp(log_estimated_duration));
    }

    // Computes the likelihoods w.r.t. the emission model.
    cube HSMM::computeEmissionsLikelihood(const mat obs) {
        return emission_->likelihoodCube(min_duration_, ndurations_, obs);
    }

    // Computes the loglikelihoods w.r.t. the emission model.
    cube HSMM::computeEmissionsLogLikelihood(const mat obs) {
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

    /**
     * Viterbi path reconstruction implementation.
     */
    void viterbiPath(const imat& psi_d, const imat& psi_s, const mat& delta,
            ivec& hiddenStates, ivec& hiddenDurations) {
        int nstates = delta.n_rows;
        int nobs = delta.n_cols;
        int curr_state = 0;
        int curr_obs_idx = nobs - 1;
        for(int i = 0; i < nstates; i++)
            if (delta(i, curr_obs_idx) > delta(curr_state, curr_obs_idx))
                curr_state = i;
        vector<int> statesSeq, durationSeq;
        while(curr_obs_idx >= 0) {
            int d = psi_d(curr_state, curr_obs_idx);
            int next_state = psi_s(curr_state, curr_obs_idx);

            // Making sure that the Viterbi algorithm ran correctly.
            assert(d >= 1);

            // current segment:
            // [curr_obs_idx - d + 1, curr_obs_idx] -> curr_state.
            statesSeq.push_back(curr_state);
            durationSeq.push_back(d);
            curr_obs_idx = curr_obs_idx - d;
            curr_state = next_state;
        }
        reverse(statesSeq.begin(), statesSeq.end());
        reverse(durationSeq.begin(), durationSeq.end());
        hiddenStates = conv_to<ivec>::from(statesSeq);
        hiddenDurations = conv_to<ivec>::from(durationSeq);
    }

};