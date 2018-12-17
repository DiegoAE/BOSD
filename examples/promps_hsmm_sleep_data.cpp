#include <armadillo>
#include <Multivariate_Gaussian_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace std;

int main(int argc, char *argv[]) {
    int nstates = 5;
    int ndimension = 10;
    vector<vec> means;
    vector<mat> covariances;
    for(int i = 0; i < nstates; i++) {
        means.push_back(zeros<vec>(ndimension));
        covariances.push_back(eye<mat>(ndimension, ndimension));
    }
    MultivariateGaussianEmission emission(means, covariances);
    cout << "OK" << endl;
    return 0;
}
