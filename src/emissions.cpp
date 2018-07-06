#include <armadillo>
#include <emissions.hpp>
#include <ForwardBackward.hpp>
#include <cmath>
#include <json.hpp>
#include <iostream>
#include <vector>

using namespace arma;
using namespace std;
using json = nlohmann::json;

namespace hsmm {

    double gaussianlogpdf_(double x, double mu, double sigma) {
        double ret = ((x - mu)*(x - mu)) / (-2*sigma*sigma);
        ret = ret - log(sqrt(2 * M_PI) * sigma);
        return ret;
    }

    double gaussianpdf_(double x, double mu, double sigma) {
        return exp(gaussianlogpdf_(x, mu, sigma));
    }

    field<mat> fromMatToField(const mat& obs) {
        field<mat> ret(obs.n_cols);
        for(int i = 0; i < obs.n_cols; i++)
            ret(i) = obs.col(i);
        return ret;
    }

    mat fromFieldToMat(const field<mat>& obs) {
        int total_cols = 0;
        int total_rows = obs(0).n_rows;
        for(auto& m : obs) {
            total_cols += m.n_cols;
            assert(m.n_rows == total_rows);
        }
        mat ret(total_rows, total_cols);
        int idx = 0;
        for(auto& m: obs) {
            ret.cols(idx, m.n_cols - 1) = m;
            idx += m.n_cols;
        }
        return ret;
    }


    /**
     * Abstract emission implementation.
     */
    AbstractEmission::AbstractEmission(int nstates, int dimension) :
            nstates_(nstates), dimension_(dimension) {}

    int AbstractEmission::getNumberStates() const {
        return nstates_;
    }

    int AbstractEmission::getDimension() const {
        return dimension_;
    }

    cube AbstractEmission::likelihoodCube(int min_duration, int ndurations,
            const field<mat>& obs) const {
        return exp(loglikelihoodCube(min_duration, ndurations, obs));
    }

    // This should return a cube of dimensions (nstates, nobs, ndurations)
    // where the entry (i, j, k) is the log-likelihood of the observations
    // in the interval [j, min_duration + k - 1] being produced by state i.
    cube AbstractEmission::loglikelihoodCube(int min_duration, int ndurations,
            const field<mat>& obs) const {
        int nobs = obs.n_elem;
        cube pdf(getNumberStates(), nobs, ndurations);
        pdf.fill(-datum::inf);
        for(int i = 0; i < getNumberStates(); i++)
            for(int t = 0; t < nobs; t++)
                for(int d = 0; d < ndurations; d++) {
                    if (t + min_duration + d > nobs)
                        break;
                    int end_idx = t + min_duration + d - 1;
                    pdf(i, t, d) = loglikelihood(i, obs.rows(t, end_idx));
                }
        return pdf;
    }

    json AbstractEmission::to_stream() const {
        cout << "Warning: serialization of emission parameters not implemented"
                << endl;
        return json::object();  // empty json object by default.
    }

    void AbstractEmission::from_stream(const json& emission_params) {
        cout << "Warning: not updating the emission parameters from the stream"
                << endl;
        return;
    }


    /**
     * DummyGaussianEmission implementation.
     */
    DummyGaussianEmission::DummyGaussianEmission(vec& means, vec& std_devs) :
            AbstractEmission(means.n_elem, 1), means_(means),
            std_devs_(std_devs) {
        assert(means_.n_elem == std_devs_.n_elem);
    }

    DummyGaussianEmission* DummyGaussianEmission::clone() const {
        return new DummyGaussianEmission(*this);
    }

    double DummyGaussianEmission::loglikelihood(int state,
            const field<mat>& obs) const {
        double ret = 0;
        for(const auto& m : obs) {
            assert(m.n_rows == 1 && m.n_cols == 1);
            ret += gaussianlogpdf_(m(0, 0), means_(state),
                    std_devs_(state));
        }
        return ret;
    }

    json DummyGaussianEmission::to_stream() const {
        vector<double> means = conv_to<vector<double>>::from(means_);
        vector<double> std_devs = conv_to<vector<double>>::from(std_devs_);
        json ret;
        ret["means"] = means;
        ret["std_devs"] = std_devs;
        return ret;
    }

    void DummyGaussianEmission::reestimate(int min_duration,
            const field<cube>& meta, const field<field<mat>>& mobs) {
        int nseq = mobs.n_elem;
        for(int i = 0; i < getNumberStates(); i++) {

            // Reestimating the mean.
            vector<double> num_mult;
            vector<double> num_obs;
            for(int s = 0; s < nseq; s++) {
                auto& obs = mobs(s);
                int nobs = obs.n_elem;
                const cube& eta = meta(s);
                int ndurations = eta.n_cols;
                for(int t = min_duration - 1; t < nobs; t++) {
                    for(int d = 0; d < ndurations; d++) {
                        int first_idx_seg = t - min_duration - d + 1;
                        if (first_idx_seg < 0)
                            break;

                        // Since the observations factorize given t, d and i.
                        for(int k = first_idx_seg; k <= t; k++) {
                            num_mult.push_back(eta(i, d, t));
                            num_obs.push_back(obs(k)(0,0));
                        }
                    }
                }
            }
            vec num_mult_v(num_mult);
            vec num_obs_v(num_obs);
            num_mult_v = num_mult_v - logsumexp(num_mult_v);
            num_mult_v = exp(num_mult_v);
            double new_mean = dot(num_mult_v, num_obs_v);

            // Reestimating the variance.
            vector<double> num_obs_var;
            for(int s = 0; s < nseq; s++) {
                const auto& obs = mobs(s);
                int nobs = obs.n_elem;
                const cube& eta = meta(s);
                int ndurations = eta.n_cols;
                for(int t = min_duration - 1; t < nobs; t++) {
                    for(int d = 0; d < ndurations; d++) {
                        int first_idx_seg = t - min_duration - d + 1;
                        if (first_idx_seg < 0)
                            break;

                        // Since the observations factorize given t, d and i.
                        for(int k = first_idx_seg; k <= t; k++) {
                            double diff = (obs(k)(0,0) - new_mean);
                            num_obs_var.push_back(diff * diff);
                        }
                    }
                }
            }
            vec num_obs_var_v(num_obs_var);
            double new_variance = dot(num_mult_v, num_obs_var_v);

            means_(i) = new_mean;
            std_devs_(i) = sqrt(new_variance);
        }
    }

    field<mat> DummyGaussianEmission::sampleFromState(int state,
            int size) const {
        mat ret = randn<mat>(1, size) * std_devs_(state) + means_(state);
        return fromMatToField(ret);
    }


    /**
     * DummyMultivariateGaussianEmission implementation.
     */
    DummyMultivariateGaussianEmission::DummyMultivariateGaussianEmission(
            mat& means, double std_dev_output_noise) :
            AbstractEmission(means.n_rows, means.n_cols), means_(means),
            std_dev_output_noise_(std_dev_output_noise) {}

    DummyMultivariateGaussianEmission* DummyMultivariateGaussianEmission::
            clone() const {
        return new DummyMultivariateGaussianEmission(*this);
    }

    double DummyMultivariateGaussianEmission::loglikelihood(int state,
                const field<mat>& obs) const {
        mat copy_obs = fromFieldToMat(obs);
        assert(copy_obs.n_rows == getDimension());
        int size = copy_obs.n_cols;
        for(int i = 0; i < getDimension(); i++)
            copy_obs.row(i) -= linspace<rowvec>(0.0, 1.0, size) +
                    means_(state, i);
        double ret = 0.0;
        for(int i = 0; i < getDimension(); i++)
            for(int j = 0; j < size; j++)
                ret += gaussianlogpdf_(copy_obs(i, j), 0,
                        std_dev_output_noise_);
        return ret;
    }

    void DummyMultivariateGaussianEmission::reestimate(int min_duration,
            const field<cube>& eta, const field<field<mat>>& mobs) {
        // TODO.
    }

    field<mat> DummyMultivariateGaussianEmission::sampleFromState(
            int state, int size) const {
        mat ret = randn<mat>(getDimension(), size) * std_dev_output_noise_;
        for(int i = 0; i < getDimension(); i++)
            ret.row(i) += linspace<rowvec>(0.0, 1.0, size) + means_(state, i);
        return fromMatToField(ret);
    }
};

