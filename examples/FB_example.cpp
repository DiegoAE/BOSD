#include <armadillo>
#include <ForwardBackward.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace arma;
using namespace std;

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

class AbstractEmission {
    public:
        AbstractEmission(int nstates, int dimension) : nstates_(nstates),
                dimension_(dimension) {}

        int getNumberStates() const {
            return nstates_;
        }

        int getDimension() const {
            return dimension_;
        }

        virtual AbstractEmission* clone() const = 0;

        virtual double likelihood(int state, const mat& obs) const = 0;

        // This should return a cube of dimensions (nstates, nobs, ndurations)
        // where the entry (i, j, k) is the likelihood of the observations in
        // the interval [j, min_duration + k - 1] being produced by state i.
        virtual cube likelihoodCube(int min_duration, int ndurations,
                const mat& obs) const {
            int nobs = obs.n_cols;
            cube pdf(getNumberStates(), nobs, ndurations, fill::zeros);
            for(int i = 0; i < getNumberStates(); i++)
                for(int t = 0; t < nobs; t++)
                    for(int d = 0; d < ndurations; d++) {
                        if (t + min_duration + d > nobs)
                            break;
                        int end_idx = t + min_duration + d - 1;
                        pdf(i, t, d) = likelihood(i, obs.cols(t, end_idx));
                    }
            return pdf;
        }

        virtual void printParameters() const {
            return;
        }

        // Reestimates in place the emission parameters using the statistics
        // provided by the HSMM E step. eta(j, d, t) represents the expected
        // value of state j generating a segment of length min_duration + d
        // ending at time t.
        virtual void reestimate(int min_duration, const cube& eta,
                const mat& obs) = 0;

        virtual mat sampleFromState(int state, int size) const = 0;

    private:
        int nstates_;
        int dimension_;
};

class DummyGaussianEmission : public AbstractEmission {
    public:
        DummyGaussianEmission(vec& means, vec& std_devs) :
                AbstractEmission(means.n_elem, 1), means_(means),
                std_devs_(std_devs) {
            assert(means_.n_elem == std_devs_.n_elem);
        }

        DummyGaussianEmission* clone() const {
            return new DummyGaussianEmission(*this);
        }

        double likelihood(int state, const mat& obs) const {
            double ret = 1.0;
            for(int i = 0; i < obs.n_cols; i++)
                ret *= gaussianpdf_(obs(0, i), means_(state),
                        std_devs_(state));
            return ret;
        }

        virtual void printParameters() const {
            cout << "means:" << endl << means_ << endl;
            cout << "std_devs:" << endl << std_devs_ << endl;
        }

        void reestimate(int min_duration, const cube& eta, const mat& obs) {
            int nobs = obs.n_cols;
            int ndurations = eta.n_cols;
            for(int i = 0; i < getNumberStates(); i++) {

                // Reestimating the mean.
                double new_mean_num = 0;
                double new_common_den = 0;
                for(int t = min_duration - 1; t < nobs; t++) {
                    for(int d = 0; d < ndurations; d++) {
                        int first_idx_seg = t - min_duration - d + 1;
                        if (first_idx_seg < 0)
                            break;

                        // Since the observations factorize given t, d and i.
                        for(int s = first_idx_seg; s <= t; s++) {
                            new_common_den += eta(i, d, t);
                            new_mean_num += eta(i, d, t) * obs(0, s);
                        }
                    }
                }
                double new_mean = new_mean_num / new_common_den;

                // Reestimating the variance.
                double new_var_num = 0;
                for(int t = min_duration - 1; t < nobs; t++) {
                    for(int d = 0; d < ndurations; d++) {
                        int first_idx_seg = t - min_duration - d + 1;
                        if (first_idx_seg < 0)
                            break;

                        // Since the observations factorize given t, d and i.
                        for(int s = first_idx_seg; s <= t; s++) {
                            double diff = (obs(0, s) - new_mean);
                            new_var_num += eta(i, d, t) * diff * diff;
                        }
                    }
                }

                means_(i) = new_mean;
                std_devs_(i) = sqrt(new_var_num / new_common_den);
            }
        }

        mat sampleFromState(int state, int size) const {
            return randn<mat>(1, size) * std_devs_(state) + means_(state);
        }

    private:
        vec means_;
        vec std_devs_;
};

class DummyMultivariateGaussianEmission : public AbstractEmission {
    public:
        DummyMultivariateGaussianEmission(mat& means, double std_dev_output_noise) :
                AbstractEmission(means.n_rows, means.n_cols), means_(means),
                std_dev_output_noise_(std_dev_output_noise) {}

        DummyMultivariateGaussianEmission* clone() const {
            return new DummyMultivariateGaussianEmission(*this);
        }

        double likelihood(int state, const mat& obs) const {
            assert(obs.n_rows == getDimension());
            int size = obs.n_cols;
            mat copy_obs(obs);
            for(int i = 0; i < getDimension(); i++)
                copy_obs.row(i) -= linspace<rowvec>(0.0, 1.0, size) + means_(state, i);
            double ret = 1.0;
            for(int i = 0; i < getDimension(); i++)
                for(int j = 0; j < size; j++)
                    ret *= gaussianpdf_(copy_obs(i, j), 0, std_dev_output_noise_);
            return ret;
        }

        void reestimate(int min_duration, const cube& eta, const mat& obs) {
            // TODO.
        }

        mat sampleFromState(int state, int size) const {
            mat ret = randn<mat>(getDimension(), size) * std_dev_output_noise_;
            for(int i = 0; i < getDimension(); i++)
                ret.row(i) += linspace<rowvec>(0.0, 1.0, size) + means_(state, i);
            return ret;
        }

    private:
        double std_dev_output_noise_;
        mat means_;
};

class HSMM {
    public:
        HSMM(shared_ptr<AbstractEmission> emission, mat transition,
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

        void setDuration(mat duration) {
            assert(duration.n_rows == nstates_);
            assert(duration.n_cols == ndurations_);
            duration_ = duration;
        }

        void setEmission(shared_ptr<AbstractEmission> emission) {
            assert(emission->getNumberStates() == nstates_);
            emission_ = emission;
        }

        void setPi(vec pi) {
            assert(pi.n_elem == nstates_);
            pi_ = pi;
        }

        void setTransition(mat transition) {
            assert(transition.n_rows == transition.n_cols);
            assert(transition.n_rows == nstates_);
            transition_ = transition;
        }

        mat sampleSegments(int nsegments, ivec& hiddenStates,
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

        void fit(mat obs, int max_iter, double tol) {
            // For now only the transition matrix is being learned.
            int nobs = obs.n_cols;
            assert(nobs >= 1);
            mat alpha(nstates_, nobs, fill::zeros);
            mat beta(nstates_, nobs, fill::zeros);
            mat alpha_s(nstates_, nobs, fill::zeros);
            mat beta_s(nstates_, nobs, fill::zeros);
            vec beta_s_0(nstates_, fill::zeros);
            cube eta(nstates_, ndurations_, nobs, fill::zeros);
            mat estimated_transition(transition_);
            vec estimated_pi(pi_);
            mat estimated_duration(duration_);
            double marginal_llikelihood = -datum::inf;
            bool convergence_reached = false;
            for(int i = 0; i < max_iter && !convergence_reached; i++) {
                // Recomputing the emission likelihoods.
                cube pdf = computeEmissionsLikelihood(obs);

                FB(estimated_transition, estimated_pi, estimated_duration, pdf,
                        alpha, beta, alpha_s, beta_s, beta_s_0, eta,
                        min_duration_, nobs);

                // Computing the marginal likelihood (aka observation likelihood).
                double current_llikelihood = 0.0;
                for(int j = 0; j < nstates_; j++)
                    current_llikelihood += alpha(j, nobs - 1);
                current_llikelihood = log(current_llikelihood);
                cout << "EM iteration " << i << " marginal log-likelihood: " <<
                        current_llikelihood << ". Diff: " <<
                        current_llikelihood - marginal_llikelihood << endl;
                assert(!(current_llikelihood < marginal_llikelihood));
                if (current_llikelihood - marginal_llikelihood < tol) {
                    convergence_reached = true;
                    marginal_llikelihood = current_llikelihood;
                    break;
                }
                marginal_llikelihood = current_llikelihood;

                // Reestimating transitions.
                mat tmp_transition(size(transition_), fill::zeros);
                for(int t = 0; t < nobs - 1; t++)
                    for(int i = 0; i < nstates_; i++)
                        for(int j = 0; j < nstates_; j++)
                            tmp_transition(i, j) += alpha(i, t) *
                                    estimated_transition(i, j) * beta_s(j, t);
                vec row_sums = sum(tmp_transition, 1);
                for(int r = 0; r < nstates_; r++)
                    tmp_transition.row(r) /= row_sums(r);
                estimated_transition = tmp_transition;

                // Reestimating the initial state pmf.
                estimated_pi = beta_s_0 % estimated_pi;
                // double mllh = log(sum(estimated_pi));
                estimated_pi = estimated_pi / sum(estimated_pi);

                // Reestimating durations.
                // D(j, d) represents the expected number of times that state
                // j is visited with duration d (non-normalized).
                mat D = sum(eta, 2);
                vec D_sums = sum(D, 1);
                for(int r = 0; r < nstates_; r++)
                    D.row(r) /= D_sums(r);
                estimated_duration = D;

                // Reestimating emissions.
                // NOTE: the rest of the HSMM parameters are updated out of
                // this loop.
                emission_->reestimate(min_duration_, eta, obs);
            }

            cout << "Stopped because of " << ((convergence_reached) ?
                    "convergence." : "max iter.") << endl;

            // Updating the model parameters.
           setTransition(estimated_transition);
           setPi(estimated_pi);
           setDuration(estimated_duration);
        }

        // Computes the likelihoods w.r.t. the emission model.
        cube computeEmissionsLikelihood(const mat obs) {
            return emission_->likelihoodCube(min_duration_, ndurations_, obs);
        }

        mat transition_;
        vec pi_;
        mat duration_;
        int ndurations_;
        int min_duration_;
        int nstates_;
        shared_ptr<AbstractEmission> emission_;
};

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

        // current segment: [curr_obs_idx - d + 1, curr_obs_idx] -> curr_state.
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

int main() {
    int ndurations = 4;
    int min_duration = 4;
    mat transition = {{0.0, 0.1, 0.4, 0.5},
                      {0.3, 0.0, 0.6, 0.1},
                      {0.2, 0.2, 0.0, 0.6},
                      {0.4, 0.4, 0.2, 0.0}};
    int nstates = transition.n_rows;
    vec pi(nstates, fill::eye);
    pi.fill(1.0/nstates);
    // mat durations(nstates, ndurations, fill::eye);
    mat durations =  {{0.0, 0.1, 0.4, 0.5},
                      {0.3, 0.0, 0.6, 0.1},
                      {0.2, 0.2, 0.0, 0.6},
                      {0.4, 0.4, 0.2, 0.0}};

    // Instantiating the emission process.
    vec means = {0, 5, 10, 15};
    vec std_devs =  {0.5, 1.0, 0.1, 2.0};
    shared_ptr<AbstractEmission> ptr_emission(new DummyGaussianEmission(
            means, std_devs));

    // Multivariate emission.
    mat mult_means(nstates, 2, fill::zeros);
    for(int i = 0; i < nstates; i++)
        mult_means.row(i).fill(i);
    shared_ptr<AbstractEmission> ptr_mult_emission(
            new DummyMultivariateGaussianEmission(mult_means, 0.1));
    mat mult_sample = ptr_mult_emission->sampleFromState(0, 5);

    // Instantiating the HSMM.
    HSMM dhsmm(ptr_emission, transition, pi, durations, min_duration);

    ivec hiddenStates, hiddenDurations;
    int nSampledSegments = 35;
    mat samples = dhsmm.sampleSegments(nSampledSegments, hiddenStates,
            hiddenDurations);
    int nobs = samples.n_cols;
    cube pdf = dhsmm.computeEmissionsLikelihood(samples);
    cout << "Generated samples" << endl;
    // cout << samples << endl;
    cout << "Generated states and durations" << endl;
    cout << join_horiz(hiddenStates, hiddenDurations) << endl;

    mat alpha(nstates, nobs, fill::zeros);
    mat beta(nstates, nobs, fill::zeros);
    mat alpha_s(nstates, nobs, fill::zeros);
    mat beta_s(nstates, nobs, fill::zeros);
    vec beta_s_0(nstates, fill::zeros);
    cube eta(nstates, ndurations, nobs, fill::zeros);
    FB(transition, pi, durations, pdf, alpha, beta, alpha_s, beta_s, beta_s_0,
            eta, min_duration, nobs);
    cout << "Alpha" << endl;
    // cout << alpha << endl;
    cout << "Beta" << endl;
    // cout << beta << endl;
    cout << "Alpha normalized" << endl;
    mat alpha_normalized(alpha);
    for(int i = 0; i < alpha.n_rows; i++)
        alpha_normalized.row(i) /= sum(alpha, 0);
    // cout << alpha_normalized << endl;
    cout << "Sums columns" << endl;
    // cout << sum(alpha, 0) << endl;
    // cout << sum(beta, 0) << endl;
    cout << "Sums rows" << endl;
    cout << sum(alpha, 1) << endl;
    cout << sum(beta, 1) << endl;
    imat psi_duration(nstates, nobs, fill::zeros);
    imat psi_state(nstates, nobs, fill::zeros);
    mat delta(nstates, nobs, fill::zeros);
    Viterbi(transition, pi, durations, pdf, delta, psi_duration, psi_state,
            min_duration, nobs);
    cout << "Delta last column" << endl;
    cout << delta.col(nobs - 1) << endl;
    ivec viterbiStates, viterbiDurations;
    viterbiPath(psi_duration, psi_state, delta, viterbiStates,
            viterbiDurations);

    cout << "Viterbi states and durations" << endl;
    cout << join_horiz(viterbiStates, viterbiDurations) << endl;

    // Debug
    int differences = 0;
    if (viterbiStates.n_elem == hiddenStates.n_elem) {
        for(int t = 0; t < viterbiStates.n_elem; t++)
            if (hiddenStates(t) != viterbiStates(t))
                differences++;
        cout << " Differences: " << differences << endl;
    }
    else
        cout << "The dimensions don't match." << endl;

    // Initializing uniformly the transitions, initial state pmf and durations.
    transition.fill(1.0/(nstates-1));
    transition.diag().zeros();  // No self-loops.
    dhsmm.setTransition(transition);
    pi.fill(1.0/nstates);
    dhsmm.setPi(pi);
    durations.fill(1.0/ndurations);
    dhsmm.setDuration(durations);

    // Resetting emission parameters.
    vec new_means = {0.1, 0.2, 0.3, 0.4};
    vec new_std_devs = ones<vec>(nstates) * 10;
    shared_ptr<AbstractEmission> init_emission(new DummyGaussianEmission(
            new_means, new_std_devs));
    dhsmm.setEmission(init_emission);

    cout << "Best transition matrix we can aim at:" << endl;
    mat prueba(nstates, nstates, fill::zeros);
    for(int i = 0; i < hiddenStates.n_elem - 1; i++)
        prueba(hiddenStates(i), hiddenStates(i + 1))++;
    mat pruebasum = sum(prueba, 1);
    for(int i = 0; i < nstates; i++)
        prueba.row(i) /= pruebasum(i);
    cout << prueba << endl;

    // Testing the learning algorithm.
    dhsmm.fit(samples, 100, 1e-10);
    cout << "Learnt matrix:" << endl;
    cout << dhsmm.transition_ << endl;

    cout << "Best duration matrix we can aim at:" << endl;
    mat emp_durations(nstates, ndurations, fill::zeros);
    for(int i = 0; i < hiddenStates.n_elem; i++)
        emp_durations(hiddenStates(i), hiddenDurations(i) - min_duration)++;
    mat emp_durations_sum = sum(emp_durations, 1);
    for(int i = 0; i < nstates; i++)
        emp_durations.row(i) /= emp_durations_sum(i);
    cout << emp_durations << endl;
    cout << "Learnt durations:" << endl;
    cout << dhsmm.duration_ << endl;

    cout << "Learnt pi:" << endl;
    cout << dhsmm.pi_ << endl;

    cout << "Learnt emission parameters" << endl;
    dhsmm.emission_->printParameters();
    return 0;
}