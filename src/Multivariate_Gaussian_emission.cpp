#include <armadillo>
#include <cmath>
#include <json.hpp>
#include <iostream>
#include <Multivariate_Gaussian_emission.hpp>
#include <vector>

using namespace arma;
using namespace std;
using json = nlohmann::json;   
 
namespace hsmm {

    /**
     * MultivariateGaussianEmission implementation.
     */
    MultivariateGaussianEmission::MultivariateGaussianEmission(
            vector<vec> means, vector<mat> covs) : means_(means), covs_(covs),
            AbstractEmissionObsCondIIDgivenState(means.size(),
            means.at(0).n_elem) {
        assert(means_.size() == covs_.size());
        for(auto& v: means_)
            assert(v.n_elem == dimension_);
        for(auto& m: covs_)
            assert(m.n_rows == dimension_ && m.n_cols == dimension_);
    }

    double MultivariateGaussianEmission::loglikelihood(int state,
            const mat &single_obs) const {
        //TODO
        return 0.0;
    }

    void MultivariateGaussianEmission::fitFromLabels(field<mat> &observations,
            ivec &labels) {
        assert(observations.n_elem == labels.n_elem);
        ivec counts(nstates_, fill::zeros);
        vector<vec> new_means(means_);
        vector<mat> new_covs(covs_);
        for(int i = 0; i < nstates_; i++) {
            new_means.at(i).zeros();
            new_covs.at(i).zeros();
        }
        for(int i = 0; i < labels.n_elem; i++) {
            auto &x = observations(i);
            new_means.at(labels(i)) += x;
            new_covs.at(labels(i)) += x * x.t();
            counts(labels(i)) += 1;
        }
        for(int i = 0; i < nstates_; i++) {
            assert(counts(i) > 0);
            new_means.at(i) = new_means.at(i) / counts(i);
            new_covs.at(i) = (new_covs.at(i) / counts(i)) -
                    new_means.at(i) * new_means.at(i).t();
        }
        means_ = new_means;
        covs_ = new_covs;
        return;
    }
};

