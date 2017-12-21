#include <armadillo>

void FB(const arma::mat& transition,const arma::vec& pi,
        const arma::mat& duration, const arma::cube& pdf, arma::mat& alpha,
        arma::mat& beta, arma::mat& alpha_s, arma::mat& beta_s,
        arma::vec& beta_s_0, const int min_duration, const int nobs);

void Viterbi(const arma::mat& transition,const arma::vec& pi,
        const arma::mat& duration, const arma::cube& pdf, arma::mat& delta,
        arma::imat& psi_duration, arma::imat& psi_state,
        const int min_duration, const int nobs);
