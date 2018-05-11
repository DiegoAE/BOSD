#include <algorithm>
#include <ForwardBackward.hpp>
#include <cassert>
#include <vector>

#define EPS 1e-7

using namespace arma;
using namespace std;

void myassert(bool condition) {
    if (!condition)
        throw std::logic_error("Assertion failed");
}

void myassert(bool condition, string message) {
    if (!condition)
        throw std::logic_error(message);
}

bool equal(double a, double b) {
    return fabs(a - b) < EPS;
}

double logsumexp(vec c) {
    // computes log(sum_ i(exp(x_i))) with the so called log-sum-exp trick.
    double maxv = c.max();

    // Handling the special case of all being -inf.
    if (maxv == -datum::inf)
        return maxv;

    double sum = 0.0;
    for(int i = 0; i < c.n_elem; i++)
        sum += exp(c(i) - maxv);
    return maxv + log(sum);
}

void safety_checks(const mat& transition,const vec& pi, const mat& duration,
        const cube& pdf, const mat& alpha, const mat& beta,
        const mat& alpha_s, const mat& beta_s, const int min_duration,
        const int nobs) {
    int nstates = transition.n_rows;
    int duration_steps = duration.n_cols;
    // Dimensions checking.
    assert(transition.n_rows == transition.n_cols);
    assert(nstates == pi.n_elem);
    assert(nstates == duration.n_rows);
    assert(nstates == pdf.n_rows);
    assert(pdf.n_cols == nobs);
    assert(duration_steps == pdf.n_slices);
    assert(min_duration >= 1);
    assert(nobs >= min_duration);
    assert(alpha.n_rows == nstates);
    assert(alpha.n_cols == nobs);
    assert(beta.n_rows == nstates);
    assert(beta.n_cols == nobs);
    assert(alpha_s.n_rows == nstates);
    assert(alpha_s.n_cols == nobs);
    assert(beta_s.n_rows == nstates);
    assert(beta_s.n_cols == nobs);
    // Normalization checking.
    vec row_sums = sum(transition, 1);
    for(int i = 0; i < transition.n_rows; i++)
        assert(equal(row_sums(i), 1.0) || equal(row_sums(i), 0.0));
    assert(equal(sum(pi), 1.0));
    vec duration_row_sums = sum(duration, 1);
    for(int i = 0; i < duration.n_rows; i++)
        assert(equal(duration_row_sums(i), 1.0) ||
                equal(duration_row_sums(i), 0.0));
}

void log_safety_checks(const mat& log_transition,const vec& log_pi,
        const mat& log_duration, const cube& pdf, const mat& alpha,
        const mat& beta, const mat& alpha_s, const mat& beta_s,
        const int min_duration, const int nobs) {
    safety_checks(exp(log_transition), exp(log_pi), exp(log_duration), pdf,
            alpha, beta, alpha_s, beta_s, min_duration, nobs);
}

void Debug(char c, int a, int b, int d) {
    cout << c << "[" << a << "," << b << "]" << " " << d << endl;
}

double log_duration_t(double actual_value, int hidden_state, int d, int end_t,
        const Labels& obs_segments) {
    if (obs_segments.isLabel(end_t, d, hidden_state) ||
            obs_segments.isLabel(end_t, d))
        return actual_value;
    if (!obs_segments.isConsistent(end_t, d, hidden_state))
        return -datum::inf;
    return actual_value;
}

double log_transition_t(double actual_value, int hs_from, int hs_to,
        int end_t, const Labels& obs_segments) {
    if (obs_segments.transition(hs_from, hs_to, end_t))
        return actual_value;
    if (obs_segments.transition(end_t))
        return -datum::inf;
    return actual_value;
}

double log_pi_t(double actual_value, int hidden_state, int d,
        const Labels& obs_segments) {
    if (obs_segments.empty())
        return actual_value;
    const ObservedSegment& first = obs_segments.getFirstSegment();
    if (first.getStartingTime() == 0) {
        if (first.getDuration() == d && first.getHiddenState() == hidden_state)
            return actual_value;
        else
            return -datum::inf;
    }
    return actual_value;
}

void FB(const mat& transition,const vec& pi, const mat& duration,
        const cube& pdf, const Labels& obs_segments, mat& alpha, mat& beta,
        mat& alpha_s, mat& beta_s, vec& beta_s_0, cube& eta,
        const int min_duration, const int nobs) {
    safety_checks(transition, pi, duration, pdf, alpha, beta, alpha_s, beta_s,
            min_duration, nobs);
    int nstates = transition.n_rows;
    int duration_steps = duration.n_cols;
    alpha.fill(0.0);
    beta.fill(0.0);
    alpha_s.fill(0.0);
    beta_s.fill(0.0);
    // Forward recursion.
    for(int t = min_duration - 1; t < nobs; t++) {
        for(int j = 0; j < nstates; j++) {
            alpha(j, t) = 0;
            for(int d = 0; d < duration_steps; d++) {
                int first_idx_seg = t - min_duration - d + 1;
                if (first_idx_seg < 0)
                    break;
                // Debug('f', first_idx_seg, t, first_idx_seg - 1);
                double e_lh = pdf(j, first_idx_seg, d) * duration(j, d);
                double sum = 0;
                if (first_idx_seg == 0)
                    sum = pi(j);  // Forward pass base case.
                else {
                    for(int i = 0; i < nstates; i++)
                        sum += transition(i, j) * alpha(i, first_idx_seg - 1);
                    alpha_s(j, first_idx_seg - 1) = sum;
                }
                alpha(j, t) += e_lh * sum;
            }
        }
    }
    // Backward pass base case.
    for(int i = 0; i < nstates; i++)
        beta(i, nobs - 1) = beta_s(i, nobs - 1) = 1.0;
    // Backward recursion.
    for(int t = nobs - min_duration - 1; t >= 0; t--) {
        for(int j = 0; j < nstates; j++) {
            beta(j, t) = 0;
            for(int i = 0; i < nstates; i++) {
                double sum = 0;
                for(int d = 0; d < duration_steps; d++) {
                    int first_idx_seg = t + 1;
                    int last_idx_seg = t + min_duration + d;
                    if (last_idx_seg >= nobs)
                        break;
                    // Debug('b', first_idx_seg, last_idx_seg, last_idx_seg);
                    double e_lh = pdf(i, first_idx_seg, d) * duration(i, d);
                    sum += e_lh * beta(i, last_idx_seg);
                }
                beta_s(i, t) = sum;
                beta(j, t) += transition(j, i) * sum;
            }
        }
    }

    // Computing beta*_0(j) required to estimate pi.
    beta_s_0 = zeros<vec>(nstates);
    for(int j = 0; j < nstates; j++) {
        for(int d = 0; d < duration_steps; d++) {
            int first_idx_seg = 0;
            int last_idx_seg = min_duration + d - 1;
            if (last_idx_seg >= nobs)
                break;
            double e_lh = pdf(j, first_idx_seg, d) * duration(j, d);
            beta_s_0(j) += e_lh * beta(j, last_idx_seg);
        }
    }

    // Computing eta(j, d, t). The expected value of state j generating a
    // segment of length min_duration + d ending at time t (non-normalized).
    eta = zeros<cube>(nstates, duration_steps, nobs);
    for(int t = min_duration - 1; t < nobs; t++) {
        for(int j = 0; j < nstates; j++) {
            for(int d = 0; d < duration_steps; d++) {
                int first_idx_seg = t - min_duration - d + 1;
                if (first_idx_seg < 0)
                    break;
                double e_lh = pdf(j, first_idx_seg, d) * duration(j, d);
                double left_side = (first_idx_seg == 0) ?
                        pi(j) : alpha_s(j, first_idx_seg - 1);
                double right_side = beta(j, t);
                eta(j, d, t) = left_side * e_lh * right_side;
            }
        }
    }
}

void logsFB(const arma::mat& log_transition,const arma::vec& log_pi,
        const arma::mat& log_duration, const arma::cube& log_pdf,
        const Labels& obs_segments, arma::mat& alpha,
        arma::mat& beta, arma::mat& alpha_s, arma::mat& beta_s,
        arma::vec& beta_s_0, arma::cube& eta, arma::cube &zeta,
        const int min_duration, const int nobs) {
    log_safety_checks(log_transition, log_pi, log_duration, log_pdf, alpha,
            beta, alpha_s, beta_s, min_duration, nobs);
    int nstates = log_transition.n_rows;
    int duration_steps = log_duration.n_cols;
    alpha.fill(-datum::inf);
    alpha_s.fill(-datum::inf);
    beta.fill(-datum::inf);
    beta_s.fill(-datum::inf);
    // Forward recursion.
    for(int t = min_duration - 1; t < nobs; t++) {
        for(int j = 0; j < nstates; j++) {
            vec c_alpha(duration_steps);
            c_alpha.fill(-datum::inf);
            for(int d = 0; d < duration_steps; d++) {
                int first_idx_seg = t - min_duration - d + 1;
                if (first_idx_seg < 0)
                    break;
                // Debug('f', first_idx_seg, t, first_idx_seg - 1);
                double e_lh = log_pdf(j, first_idx_seg, d) +
                        log_duration_t(log_duration(j, d), j, d + min_duration,
                        t, obs_segments);
                vec c_alpha_s;
                if (first_idx_seg == 0)
                    c_alpha_s = log_pi_t(log_pi(j), j, d + min_duration, obs_segments);
                else {
                    c_alpha_s = zeros<vec>(nstates);
                    for(int i = 0; i < nstates; i++)
                        c_alpha_s(i) = log_transition_t(log_transition(i, j),
                                i, j, first_idx_seg - 1, obs_segments) +
                                alpha(i, first_idx_seg - 1);
                    alpha_s(j, first_idx_seg - 1) = logsumexp(c_alpha_s);
                }
                c_alpha(d) = e_lh + logsumexp(c_alpha_s);
            }

            alpha(j, t) = logsumexp(c_alpha);
        }
    }

    // Backward pass base case.
    for(int i = 0; i < nstates; i++)
        beta(i, nobs - 1) = beta_s(i, nobs - 1) = 0.0;
    // Backward recursion.
    for(int t = nobs - min_duration - 1; t >= 0; t--) {
        for(int j = 0; j < nstates; j++) {
            vec c_beta(nstates);
            c_beta.fill(-datum::inf);
            for(int i = 0; i < nstates; i++) {
                vec c_beta_s(duration_steps);
                c_beta_s.fill(-datum::inf);
                for(int d = 0; d < duration_steps; d++) {
                    int first_idx_seg = t + 1;
                    int last_idx_seg = t + min_duration + d;
                    if (last_idx_seg >= nobs)
                        break;
                    // Debug('b', first_idx_seg, last_idx_seg, last_idx_seg);
                    double e_lh = log_pdf(i, first_idx_seg, d) +
                            log_duration_t(log_duration(i, d), i,
                            d + min_duration, last_idx_seg, obs_segments);
                    c_beta_s(d) = e_lh + beta(i, last_idx_seg);
                }
                beta_s(i, t) = logsumexp(c_beta_s);
                c_beta(i) = log_transition_t(log_transition(j, i), j, i, t,
                        obs_segments) + beta_s(i, t);
            }
            beta(j, t) = logsumexp(c_beta);
        }
    }

    // Computing beta*_0(j) required to estimate pi.
    beta_s_0 = zeros<vec>(nstates);
    beta_s_0.fill(-datum::inf);
    for(int j = 0; j < nstates; j++) {
        vec c_beta_s_0(duration_steps);
        c_beta_s_0.fill(-datum::inf);
        for(int d = 0; d < duration_steps; d++) {
            int first_idx_seg = 0;
            int last_idx_seg = min_duration + d - 1;
            if (last_idx_seg >= nobs)
                break;
            double e_lh = log_pdf(j, first_idx_seg, d) +
                    log_duration_t(log_duration(j, d), j,
                    d + min_duration, last_idx_seg, obs_segments);
            c_beta_s_0(d) = e_lh + beta(j, last_idx_seg);
        }
        beta_s_0(j) = logsumexp(c_beta_s_0);
    }

    // Computing eta(j, d, t). The expected value of state j generating a
    // segment of length min_duration + d ending at time t (non-normalized).
    eta = zeros<cube>(nstates, duration_steps, nobs);
    eta.fill(-datum::inf);
    for(int t = min_duration - 1; t < nobs; t++) {
        for(int j = 0; j < nstates; j++) {
            for(int d = 0; d < duration_steps; d++) {
                int first_idx_seg = t - min_duration - d + 1;
                if (first_idx_seg < 0)
                    break;
                double e_lh = log_pdf(j, first_idx_seg, d) +
                        log_duration_t(log_duration(j, d), j, d + min_duration,
                        t, obs_segments);
                double left_side = (first_idx_seg == 0) ?
                        log_pi_t(log_pi(j), j, d + min_duration, obs_segments) :
                        alpha_s(j, first_idx_seg - 1);
                double right_side = beta(j, t);
                eta(j, d, t) = left_side + e_lh + right_side;
            }
        }
    }

    // Computing zeta(i, j, t). The expected value of having a transition from
    // state i to state j at time t (non-normalized).
    zeta = zeros<cube>(nstates, nstates, nobs - 1);
    for(int i = 0; i < nstates; i++) {
        for(int j = 0; j < nstates; j++) {
            for(int t = 0; t < nobs - 1; t++) {
                zeta(i, j, t) = alpha(i, t) + log_transition_t(
                        log_transition(i, j), i, j, t, obs_segments) +
                        beta_s(j, t);
            }
        }
    }

    // Normalizing eta and zeta.
    double llikelihood = logsumexp(alpha.col(nobs - 1));
    eta -= llikelihood;
    zeta -= llikelihood;
}

/**
 * Viterbi algorithm to find the most likely segmentation and hidden state
 * correspondence. It's implemented with the log-scaled parameters for numerical
 * stability.
 */
void Viterbi(const mat& transition,const vec& pi, const mat& duration,
        const cube& log_pdf, mat& delta, imat& psi_duration, imat& psi_state,
        const int min_duration, const int nobs) {
    mat log_transition = log(transition);
    mat log_pi = log(pi);
    mat log_duration = log(duration);
    log_safety_checks(log_transition, log_pi, log_duration, log_pdf, delta,
            delta, conv_to<mat>::from(psi_duration),
            conv_to<mat>::from(psi_state), min_duration, nobs);
    int nstates = transition.n_rows;
    int duration_steps = duration.n_cols;
    delta.fill(-datum::inf);
    psi_duration.fill(-1);
    psi_state.fill(-1);
    for(int t = min_duration - 1; t < nobs; t++) {
        for(int j = 0; j < nstates; j++) {
            delta(j, t) = -datum::inf;
            int best_duration = -1;
            int best_state = -1;
            for(int d = 0; d < duration_steps; d++) {
                int first_idx_seg = t - min_duration - d + 1;
                if (first_idx_seg < 0)
                    break;
                double e_lh =log_pdf(j, first_idx_seg, d) + log_duration(j, d);
                if (first_idx_seg == 0) {
                    // Base case.
                    if (e_lh + log_pi(j) > delta(j, t)) {
                        delta(j, t) = e_lh + log_pi(j);
                        best_duration = min_duration + d;
                        best_state = -1;  // There isn't a previous state.
                    }
                }
                else {
                    for(int i = 0; i < nstates; i++) {
                        double tmp = e_lh + log_transition(i, j);
                        tmp += delta(i, first_idx_seg - 1);
                        if (tmp > delta(j, t)) {
                            delta(j, t) = tmp;
                            best_duration = min_duration + d;
                            best_state = i;
                        }
                    }
                }
            }
            psi_duration(j, t) = best_duration;
            psi_state(j, t) = best_state;
        }
    }
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

/*
 * ObservedSegment Implementation. This supports semi-supervised learning.
 */
ObservedSegment::ObservedSegment(int t, int d) :
        ObservedSegment::ObservedSegment(t, d, -1) {}

ObservedSegment::ObservedSegment(int t, int d, int hidden_state) :
        t_(t), d_(d), hidden_state_(hidden_state) {
    myassert(t >= 0);
    myassert(hidden_state >= -1);
    myassert(d_ >= 1);
    myassert(getStartingTime() >= 0);
}

int ObservedSegment::getDuration() const {
    return d_;
}

int ObservedSegment::getEndingTime() const {
    return t_;
}

int ObservedSegment::getHiddenState() const {
    return hidden_state_;
}

int ObservedSegment::getStartingTime() const {
    return t_ - d_ + 1;
}

bool ObservedSegment::operator< (const ObservedSegment & segment) const {
    return t_ < segment.getEndingTime();
}


/*
 * Labels implementation
 */
Labels::Labels() {}

void Labels::setLabel(int t, int d) {
    setLabel(t, d, -1);
}

void Labels::setLabel(int t, int d, int hidden_state) {
    ObservedSegment label(t, d, hidden_state);

    // Making sure that the segments are non-overlapping.
    myassert(!overlaps_(t, d));
    labels_.insert(label);
}

bool Labels::empty() const {
    return labels_.empty();
}

bool Labels::isLabel(int t, int d) const {
    return isLabel(t, d, -1);
}

bool Labels::isLabel(int t, int d, int hidden_state) const {
    ObservedSegment label(t, d, hidden_state);
    auto it = labels_.find(label);
    return it != labels_.end() && it->getDuration() == d &&
        it->getHiddenState() == hidden_state;
}

bool Labels::isConsistent(int t, int d, int hidden_state) const {
    if (isLabel(t, d) || isLabel(t, d, hidden_state))
        return true;
    if (overlaps_(t, d))
        return false;
    return true;
}

bool Labels::transition(int hs_from, int hs_to, int t) const {
    ObservedSegment label(t, 1);  // Dummy duration value.
    auto from = labels_.lower_bound(label);
    auto to = labels_.upper_bound(label);
    if (from == labels_.end() || to == labels_.end())
        return false;
    return from->getEndingTime() == t && to->getStartingTime() == (t + 1) &&
        from->getHiddenState() == hs_from && to->getHiddenState() == hs_to;
}

bool Labels::transition(int t) const {
    ObservedSegment label(t, 1);  // Dummy duration value.
    auto from = labels_.lower_bound(label);
    auto to = labels_.upper_bound(label);
    if (from == labels_.end() || to == labels_.end())
        return false;
    return from->getEndingTime() == t && to->getStartingTime() == (t + 1);
}

const ObservedSegment& Labels::getFirstSegment() const {
    myassert(!empty(), "empty segment");
    auto it = labels_.begin();
    return *it;
}

const set<ObservedSegment>& Labels::getLabels() const {
    return labels_;
}

bool Labels::overlaps_(int t, int d) const {
    ObservedSegment label(t, d);
    if (!labels_.empty()) {

        if (labels_.count(label) != 0)
            return true;
        auto right = labels_.lower_bound(label);
        if (right != labels_.end() &&
                right->getStartingTime() <= label.getEndingTime())
            return true;
        if (right != labels_.begin() &&
                (--right)->getEndingTime() >= label.getStartingTime())
            return true;
    }
    return false;
}

