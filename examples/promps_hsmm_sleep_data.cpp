#include <armadillo>
#include <Multivariate_Gaussian_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics::random;
using namespace std;

int main(int argc, char *argv[]) {
    int nstates = 5;
    int ndimension = 10;
    vector<NormalDist> states;
    for(int i = 0; i < nstates; i++) {
        vec mean = ones<vec>(ndimension) * i;
        mat cov = eye(ndimension, ndimension);
        NormalDist a(mean, cov);
        states.push_back(a);
    }
    MultivariateGaussianEmission emission(states);
    cout << "OK" << endl;
    return 0;
}
