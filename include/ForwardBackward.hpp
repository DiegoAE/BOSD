#ifndef HSMM_FORWARDBACKWARD_H
#define HSMM_FORWARDBACKWARD_H

#include <armadillo>

double logsumexp(arma::vec c);

void FB(const arma::mat& transition,const arma::vec& pi,
        const arma::mat& duration, const arma::cube& pdf, arma::mat& alpha,
        arma::mat& beta, arma::mat& alpha_s, arma::mat& beta_s,
        arma::vec& beta_s_0, arma::cube& eta, const int min_duration,
        const int nobs);

void logsFB(const arma::mat& transition,const arma::vec& pi,
        const arma::mat& duration, const arma::cube& log_pdf, arma::mat& alpha,
        arma::mat& beta, arma::mat& alpha_s, arma::mat& beta_s,
        arma::vec& beta_s_0, arma::cube& eta, const int min_duration,
        const int nobs);

void Viterbi(const arma::mat& transition,const arma::vec& pi,
        const arma::mat& duration, const arma::cube& pdf, arma::mat& delta,
        arma::imat& psi_duration, arma::imat& psi_state,
        const int min_duration, const int nobs);

void viterbiPath(const arma::imat& psi_d, const arma::imat& psi_s,
        const arma::mat& delta, arma::ivec& hiddenStates,
        arma::ivec& hiddenDurations);

#endif
