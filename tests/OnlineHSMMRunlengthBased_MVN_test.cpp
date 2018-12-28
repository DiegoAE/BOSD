#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE HSMM
#include <armadillo>
#include <boost/test/unit_test.hpp>
#include <HSMM.hpp>
#include <iostream>
#include <memory>
#include <Multivariate_Gaussian_emission.hpp>
#include <robotics/utils/random.hpp>

#define EPSILON 1e-6

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;

mat transition = {{0.0, 0.1, 0.4, 0.5},
                  {0.3, 0.0, 0.6, 0.1},
                  {0.2, 0.2, 0.0, 0.6},
                  {0.4, 0.4, 0.2, 0.0}};

mat duration = {{0.0, 0.1, 0.4, 0.5},
                {0.3, 0.0, 0.6, 0.1},
                {0.2, 0.2, 0.0, 0.6},
                {0.4, 0.4, 0.2, 0.0}};

int min_duration = 4;

vec pi = {0.25, 0.25, 0.25, 0.25};

int ndurations = duration.n_cols;

int nstates = duration.n_rows;

int ndimension = 7;

ivec expand_vit_mat(const imat& vit) {
    vector<int> ret;
    for(int i = 0; i < vit.n_rows; i++)
        for(int j = 0; j < vit(i, 1); j++)
            ret.push_back(vit(i, 0));
    return conv_to<ivec>::from(ret);
}

BOOST_AUTO_TEST_CASE(OnlineHSMMRunlengthBased_Multivariate_Gaussian_Emission) {
    vector<random::NormalDist> states;
    for(int i = 0; i < nstates; i++) {
        vec mean = ones<vec>(ndimension) * i * 10;
        mat cov = eye(ndimension, ndimension);
        random::NormalDist a(mean, cov);
        states.push_back(a);
    }
    shared_ptr<MultivariateGaussianEmission> emission(
            new MultivariateGaussianEmission(states));
    OnlineHSMMRunlengthBased model(emission, transition, pi, duration,
            min_duration);
    ivec hs, hd;
    field<mat> test_features = model.sampleSegments(50, hs, hd);
    imat vit_mat = join_horiz(hs, hd);
    mat runlength_marginals(min_duration + ndurations, test_features.n_elem);
    mat state_marginals(nstates, test_features.n_elem);
    ivec filtering_labels(test_features.n_elem);
    for(int i = 0; i < test_features.n_elem; i++) {
        model.addNewObservation(test_features.at(i));
        runlength_marginals.col(i) = model.getRunlengthMarginal();
        state_marginals.col(i) = model.getStateMarginal();
        filtering_labels(i) = (int) state_marginals.col(i).index_max();
    }
    ivec gt_labels = expand_vit_mat(vit_mat);
    BOOST_CHECK(all(gt_labels == filtering_labels));
}
