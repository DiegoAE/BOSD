#include <ForwardBackward.hpp>
#include <iostream>

using namespace arma;
using namespace std;

int main() {
    int nobs = 31;
    int nstates = 1;
    int ndurations = 3;
    int min_duration = 4;
    mat transition(nstates, nstates, fill::eye);
    vec pi = {1};
    mat durations(nstates, ndurations, fill::zeros);
    durations.col(0) = ones<vec>(nstates);
    cube pdf(nstates, nobs, ndurations, fill::zeros);
    mat alpha(nstates, nobs, fill::zeros);
    mat beta(nstates, nobs, fill::zeros);
    FB(transition, pi, durations, pdf, alpha, beta, min_duration, nobs);
    cout << alpha << endl;
    cout << beta << endl;
    return 0;
}