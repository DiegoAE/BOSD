#include <ForwardBackward.hpp>
#include <cassert>

#define EPS 1e-7

using namespace arma;

bool equal(double a, double b) {
    return fabs(a - b) < EPS;
}

void safety_checks(const mat& transition,const vec& pi, const mat& duration,
        const cube& pdf, mat& alpha, mat& beta, const int min_duration,
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
    // Normalization checking.
    assert(equal(norm(sum(transition, 1) - ones<vec>(nstates)), 0));
    assert(equal(sum(pi), 1.0));
    assert(equal(norm(sum(duration, 1) - ones<vec>(nstates)), 0));
}

void Debug(char c, int a, int b, int d) {
    cout << c << "[" << a << "," << b << "]" << " " << d << endl;
}

void FB(const mat& transition,const vec& pi, const mat& duration,
        const cube& pdf, mat& alpha, mat& beta, const int min_duration,
        const int nobs) {
    safety_checks(transition, pi, duration, pdf, alpha, beta, min_duration,
            nobs);
    int nstates = transition.n_rows;
    int duration_steps = duration.n_cols;
    for(int i = 0; i < nstates; i++)
        for(int j = 0; j < nobs; j++)
            alpha(i, j) = beta(i, j) = 0;
    // Forward recursion.
    for(int t = min_duration - 1; t < nobs; t++) {
        for(int j = 0; j < nstates; j++) {
            alpha(j, t) = 0;
            for(int d = 0; d < duration_steps; d++) {
                int first_idx_seg = t - min_duration - d + 1;
                if (first_idx_seg < 0)
                    break;
                Debug('f', first_idx_seg, t, first_idx_seg - 1);
                double e_lh = pdf(j, first_idx_seg, d) * duration(j, d);
                double sum = 0;
                if (first_idx_seg == 0)
                    sum = pi(j);  // Forward pass base case.
                else {
                    for(int i = 0; i < nstates; i++)
                        sum += transition(i, j) * alpha(i, first_idx_seg - 1);
                }
                alpha(j, t) += e_lh * sum;
            }
        }
    }
    // TODO: Include Pi(j) here and review the base case above.
    // Backward pass base case.
    for(int i = 0; i < nstates; i++)
        beta(i, nobs - 1) = 1.0;
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
                    Debug('b', first_idx_seg, last_idx_seg, last_idx_seg);
                    double e_lh = pdf(i, first_idx_seg, d) * duration(i, d);
                    sum += e_lh * beta(i, last_idx_seg);
                }
                beta(j, t) += transition(j, i) * sum;
            }
        }
    }
}