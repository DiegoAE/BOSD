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
        ProMPsEmission(vector<FullProMP> v) : AbstractEmission(promps_.size(),
                promps_.at(0).get_num_joints()), promps_(v) {
            // TODO: e.g. all have the same number of joints.
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
    // Creating a basic ProMP with the desired parameters
    vec mu_w {0, 0, 0, 0};
    mat Sigma_w = 100*eye<mat>(4,4);
    mat Sigma_y = 0.0001*eye<mat>(2,2);
    ProMP promp(mu_w, Sigma_w, Sigma_y);
    cout << "All Good." << endl;
    return 0;
}