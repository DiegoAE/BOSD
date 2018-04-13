#ifndef HSMM_FORWARDBACKWARD_H
#define HSMM_FORWARDBACKWARD_H

#include <armadillo>
#include <set>

double logsumexp(arma::vec c);

class ObservedSegment {
    public:
        ObservedSegment(int t, int d);

        ObservedSegment(int t, int d, int hidden_state);

        int getDuration() const;

        int getEndingTime() const;

        int getHiddenState() const;

        int getStartingTime() const;

        bool operator< (const ObservedSegment & segment) const;

    private:
        int t_;
        int d_;
        int hidden_state_;
};


class Labels {
    public:
        Labels();

        // Sets a segment observation ending at t with duration d.
        void setLabel(int t, int d);

        // As above but additionaly specifies the generating hidden state.
        void setLabel(int t, int d, int hidden_state);

        // Checks if a particular segment is consistent with the set labels.
        bool isConsistent(int t, int d, int hidden_state);

    private:
        bool overlaps_(int t, int d);

        std::set<ObservedSegment> labels_;
};


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
        const arma::mat& duration, const arma::cube& log_pdf, arma::mat& delta,
        arma::imat& psi_duration, arma::imat& psi_state,
        const int min_duration, const int nobs);

void viterbiPath(const arma::imat& psi_d, const arma::imat& psi_s,
        const arma::mat& delta, arma::ivec& hiddenStates,
        arma::ivec& hiddenDurations);

#endif
