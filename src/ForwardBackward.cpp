#include <ForwardBackward.hpp>
#include <cassert>

#define EPS 1e-7

using namespace arma;

bool equal(double a, double b) {
    return fabs(a - b) < EPS;
}

void safety_checks(const mat& transition,const vec& pi, const mat& duration,
        const cube& pdf, mat& alpha, const int min_duration, const int nobs) {
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
    // Normalization checking.
    assert(equal(norm(sum(transition, 1) - ones<vec>(nstates)), 0));
    assert(equal(sum(pi), 1.0));
    assert(equal(norm(sum(duration, 1) - ones<vec>(nstates)), 0));
}

void FB(const mat& transition,const vec& pi, const mat& duration,
        const cube& pdf, mat& alpha, const int min_duration, const int nobs) {
    safety_checks(transition, pi, duration, pdf, alpha, min_duration, nobs);
    int nstates = transition.n_rows;
    int duration_steps = duration.n_cols;
    for(int i = 0; i < nstates; i++)
        for(int j = 0; j < nobs; j++)
            alpha(i, j) = 0;
    // Base case.
    for(int i = 0; i < nstates; i++)
        for(int d = 0; d < duration_steps; d++) 
            alpha(i, min_duration + d) = pi(i) * pdf(i, 0, d) * duration(i, d);
    // Recursion.
    for(int t = min_duration + duration_steps; t < nobs; t++) {
        for(int j = 0; j < nstates; j++) {
            alpha[j, t] = 0;
            for(int d = 0; d < duration_steps; d++) {
                int last_change = t - min_duration - d;
                double e_lh = pdf(j, last_change + 1, d) * duration(j, d);
                double sum = 0;
                for(int i = 0; i < nstates; i++)
                    sum += transition(i, j) * alpha[i, last_change];
                alpha[j, t] += e_lh * sum;
            }
        }
    }
}