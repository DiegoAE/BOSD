#include <ForwardBackward.hpp>
#include <cassert>

#define EPS 1e-7

using namespace arma;

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
    assert(equal(norm(sum(transition, 1) - ones<vec>(nstates)), 0));
    assert(equal(sum(pi), 1.0));
    assert(equal(norm(sum(duration, 1) - ones<vec>(nstates)), 0));
}

void Debug(char c, int a, int b, int d) {
    cout << c << "[" << a << "," << b << "]" << " " << d << endl;
}

void FB(const mat& transition,const vec& pi, const mat& duration,
        const cube& pdf, mat& alpha, mat& beta, mat& alpha_s, mat& beta_s,
        vec& beta_s_0, cube& eta, const int min_duration, const int nobs) {
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

void logsFB(const arma::mat& transition,const arma::vec& pi,
        const arma::mat& duration, const arma::cube& log_pdf, arma::mat& alpha,
        arma::mat& beta, arma::mat& alpha_s, arma::mat& beta_s,
        arma::vec& beta_s_0, arma::cube& eta, const int min_duration,
        const int nobs) {
    safety_checks(transition, pi, duration, log_pdf, alpha, beta, alpha_s,
            beta_s, min_duration, nobs);
    int nstates = transition.n_rows;
    int duration_steps = duration.n_cols;
    mat log_transition = log(transition);
    mat log_duration = log(duration);
    mat log_pi = log(pi);
    alpha.fill(-datum::inf);
    alpha_s.fill(-datum::inf);
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
                        log_duration(j, d);
                vec c_alpha_s;
                if (first_idx_seg == 0)
                    c_alpha_s = log_pi(j);
                else {
                    c_alpha_s = zeros<vec>(nstates);
                    for(int i = 0; i < nstates; i++)
                        c_alpha_s(i) = log_transition(i, j) +
                                alpha(i, first_idx_seg - 1);
                    alpha_s(j, first_idx_seg - 1) = logsumexp(c_alpha_s);
                }
                c_alpha(d) = e_lh + logsumexp(c_alpha_s);
            }
            alpha(j, t) = logsumexp(c_alpha);
        }
    }
    cout << "My test" << endl;
    for(int i = 0; i < nstates; i++)
        cout << exp(logsumexp(vectorise(alpha.row(i)))) << endl;

    for(int i = 0; i < nstates; i++)
        cout << exp(logsumexp(vectorise(alpha_s.row(i)))) << endl;
}

void Viterbi(const mat& transition,const vec& pi, const mat& duration,
        const cube& pdf, mat& delta, imat& psi_duration, imat& psi_state,
        const int min_duration, const int nobs) {
    // TODO: implement logs in the Viterbi algorithm.
    safety_checks(transition, pi, duration, pdf, delta, delta,
            conv_to<mat>::from(psi_duration), conv_to<mat>::from(psi_state),
            min_duration, nobs);
    int nstates = transition.n_rows;
    int duration_steps = duration.n_cols;
    delta.fill(0.0);
    psi_duration.fill(-1);
    psi_state.fill(-1);
    for(int t = min_duration - 1; t < nobs; t++) {
        for(int j = 0; j < nstates; j++) {
            delta(j, t) = 0.0;
            int best_duration = -1;
            int best_state = -1;
            for(int d = 0; d < duration_steps; d++) {
                int first_idx_seg = t - min_duration - d + 1;
                if (first_idx_seg < 0)
                    break;
                double e_lh = pdf(j, first_idx_seg, d) * duration(j, d);
                if (first_idx_seg == 0) {
                    // Base case.
                    if (e_lh * pi(j) > delta(j, t)) {
                        delta(j, t) = e_lh * pi(j);
                        best_duration = min_duration + d;
                        best_state = -1;  // There isn't a previous state.
                    }
                }
                else {
                    for(int i = 0; i < nstates; i++) {
                        double tmp = e_lh * transition(i, j);
                        tmp *= delta(i, first_idx_seg - 1);
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
