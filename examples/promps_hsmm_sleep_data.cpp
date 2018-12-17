#include <armadillo>
#include <Multivariate_Gaussian_emission.hpp>
#include <random>

using namespace arma;
using namespace hsmm;
using namespace robotics::random;
using namespace std;

int main(int argc, char *argv[]) {
    mt19937 gen(0);
    int nstates = 5;
    int ndimension = 10;
    vector<NormalDist> states;
    vector<vec> samples;
    vector<int> labels;
    for(int i = 0; i < nstates; i++) {
        vec mean = ones<vec>(ndimension) * i;
        mat cov = eye(ndimension, ndimension);
        NormalDist a(mean, cov);
        states.push_back(a);

        // Generating toy data.
        int nsamples = 1000;
        vector<vec> s = sample_multivariate_normal(gen, a, nsamples);
        vector<int> l = conv_to<vector<int>>::from(ones<ivec>(nsamples) * i);
        samples.insert(samples.end(), s.begin(), s.end());
        labels.insert(labels.end(), l.begin(), l.end());
    }
    MultivariateGaussianEmission emission(states);
    ivec labels_vec = conv_to<ivec>::from(labels);
    field<vec> obs_field(samples.size());
    for(int i = 0; i < samples.size(); i++)
        obs_field(i) = samples.at(i);
    emission.fitFromLabels(obs_field, labels_vec);
    return 0;
}
