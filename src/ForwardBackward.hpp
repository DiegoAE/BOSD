#include <armadillo>

void FB(const arma::mat& transition,const arma::vec& pi,
        const arma::mat& duration, const arma::cube& pdf, arma::mat& alpha,
        arma::mat& beta, arma::mat& alpha_s, arma::mat& beta_s,
        const int min_duration, const int nobs);