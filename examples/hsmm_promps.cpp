#include <armadillo>
#include <HSMM.hpp>
#include <iostream>
#include <memory>
#include <robotics.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;

class ProMPsEmission : public AbstractEmission {
    public:
        ProMPsEmission(vector<FullProMP> promps) : AbstractEmission(
                promps.size(), promps.at(0).get_num_joints()),
                promps_(promps) {
            // TODO
            for(int i = 0; i < getNumberStates(); i++)
                assert(promps_.at(i).get_num_joints() == getDimension());
        }

        ProMPsEmission* clone() const {
            return new ProMPsEmission(*this);
        }

        double loglikelihood(int state, const arma::mat& obs) const {
            // TODO.
            return 0.0;
        }

        void reestimate(int min_duration, const arma::cube& eta,
                const arma::mat& obs) {
            // TODO
            return;
        }

        mat sampleFromState(int state, int size) const {
            // TODO
            return zeros<mat>(10, 10);
        }

    private:
        vector<FullProMP> promps_;
};

int main() {
    int ndurations = 4;
    int min_duration = 4;
    mat transition = {{0.0, 0.1, 0.4, 0.5},
                      {0.3, 0.0, 0.6, 0.1},
                      {0.2, 0.2, 0.0, 0.6},
                      {0.4, 0.4, 0.2, 0.0}};
    int nstates = transition.n_rows;
    vec pi(nstates, fill::eye);
    pi.fill(1.0/nstates);
    // mat durations(nstates, ndurations, fill::eye);
    mat durations =  {{0.0, 0.1, 0.4, 0.5},
                      {0.3, 0.0, 0.6, 0.1},
                      {0.2, 0.2, 0.0, 0.6},
                      {0.4, 0.4, 0.2, 0.0}};
    int n_basis_functions = 4;
    int njoints = 1;

    // Setting a third order polynomial basis function for the ProMP
    int polynomial_order = n_basis_functions - 1;
    shared_ptr<ScalarBasisFun> kernel{ new ScalarPolyBasis(polynomial_order)};

    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w(n_basis_functions);
        mu_w.fill(i);
        mat Sigma_w = 100*eye<mat>(n_basis_functions, n_basis_functions);
        mat Sigma_y = 0.0001*eye<mat>(njoints + 1, njoints + 1);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(kernel, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission.
    shared_ptr<AbstractEmission> ptr_emission(new ProMPsEmission(promps));

    HSMM promp_hsmm(ptr_emission, transition, pi, durations, min_duration);
    for(int i = 0; i < nstates; i++)
        cout << promps[i].get_model().get_mu_w() << endl;

    return 0;
}