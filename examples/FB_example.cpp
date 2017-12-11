#include <ForwardBackward.hpp>
#include <iostream>

using namespace arma;
using namespace std;

int main() {
    int nobs = 100;
    int nstates = 3;
    int ndurations = 5;
    int min_duration = 10;
    mat transition(nstates, nstates, fill::eye);
    vec pi = {1, 0, 0};
    mat durations(nstates, ndurations, fill::zeros);
    durations.col(2) = ones<vec>(nstates);
    cube pdf(nstates, nobs, ndurations, fill::zeros);
    mat alpha(nstates, nobs, fill::zeros);
    FB(transition, pi, durations, pdf, alpha, min_duration, nobs);
    cout << alpha << endl;
    return 0;
}