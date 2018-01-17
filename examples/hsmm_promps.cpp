#include <armadillo>
#include <HSMM.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <robotics.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;

// Pseudo-random number generation.
mt19937 rand_generator;

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
            const ProMP& model = promps_.at(state).get_model();
            vector<vec> w_samples = random::sample_multivariate_normal(
                    rand_generator, {model.get_mu_w(), model.get_Sigma_w()}, 1);
            vec w = w_samples.back();

            vec noise_mean = zeros<vec>(getDimension());
            vector<vec> output_noise = random::sample_multivariate_normal(
                    rand_generator, {noise_mean, model.get_Sigma_y()}, size);

            mat ret(getDimension(), size);

            // The samples are assumed to be equally spaced.
            vec sample_locations = linspace<vec>(0, 1.0, size);
            for(int i = 0; i < size; i++) {
                double z = sample_locations(i);

                // TODO: make sure this is added to the FullProMP API.
                mat phi_z = promps_.at(state).get_phi_t(z);
                ret.col(i) = phi_z * w + output_noise.at(i);
            }
            return ret;
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
        vec mu_w(n_basis_functions * njoints);
        mu_w.fill(i);
        mat Sigma_w = 100*eye<mat>(n_basis_functions * njoints,
                    n_basis_functions * njoints);
        mat Sigma_y = 0.0001*eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(kernel, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission.
    shared_ptr<AbstractEmission> ptr_emission(new ProMPsEmission(promps));

    HSMM promp_hsmm(ptr_emission, transition, pi, durations, min_duration);
    cout << ptr_emission->sampleFromState(0, 10) << endl;

    return 0;
}