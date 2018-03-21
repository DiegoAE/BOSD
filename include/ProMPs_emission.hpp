#ifndef PROMP_EMISSION_H
#define PROMP_EMISSION_H

#include <armadillo>
#include <HSMM.hpp>
#include <iostream>
#include <json.hpp>
#include <ForwardBackward.hpp>
#include <map>
#include <memory>
#include <random>
#include <robotics.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;

namespace hsmm {

    // Pseudo-random number generation.
    mt19937 rand_generator;


    class InverseWishart {
        public:
            InverseWishart(mat Phi, int dof) : Phi_(Phi), dof_(dof) {
                assert(dof > Phi.n_rows + 1);
                vec eigenvalues = eig_sym(Phi);
                assert(eigenvalues(0) > 0);
            }

            mat getPhi() {
                return Phi_;
            }

            int getDof() {
                return dof_;
            }

        private:
            mat Phi_;
            int dof_;
    };



    class ProMPsEmission : public AbstractEmission {
        public:
            ProMPsEmission(vector<FullProMP> promps) : AbstractEmission(
                    promps.size(), promps.at(0).get_num_joints()),
            promps_(promps) {
                for(int i = 0; i < getNumberStates(); i++)
                    assert(promps_.at(i).get_num_joints() == getDimension());
            }

            ProMPsEmission* clone() const {
                return new ProMPsEmission(*this);
            }

            double loglikelihood(int state, const mat& obs) const {
                const FullProMP& promp = promps_.at(state);

                // The samples are assumed to be equally spaced.
                vec sample_locations = linspace<vec>(0, 1.0, obs.n_cols);

                vec mu(promp.get_model().get_mu_w());
                mat Sigma(promp.get_model().get_Sigma_w());
                mat Sigma_y(promp.get_model().get_Sigma_y());
                double ret = 0;
                for(int i = 0; i < obs.n_cols; i++) {
                    mat Phi = promp.get_phi_t(sample_locations(i));
                    mat S = Phi * Sigma * Phi.t() + Sigma_y;

                    // Required for the marginal likelihood p(y_t | y_{1:t-1}).
                    random::NormalDist dist = random::NormalDist(Phi * mu, S);
                    ret = ret + log_normal_density(dist, obs.col(i));

                    // Using the kalman updating step to compute this efficiently.
                    mat K = Sigma * Phi.t() * inv(S);
                    mu = mu + K * (obs.col(i) - Phi * mu);
                    Sigma = Sigma - K * S * K.t();
                }
                return ret;
            }

            double loglikelihoodBatchVersion(int state, const arma::mat& obs) {
                int dimension = obs.n_rows;
                assert(dimension == getDimension());
                random::NormalDist dist = getNormalDistForMultipleTimeSteps(state,
                        obs.n_cols);
                vec stacked_obs = vectorise(obs);
                return random::log_normal_density(dist, stacked_obs);
            }

            void reestimate(int min_duration, const arma::cube& eta,
                    const arma::mat& obs) {
                int nobs = obs.n_cols;
                int ndurations = eta.n_cols;

                for(int i = 0; i < getNumberStates(); i++) {
                    ProMP promp = promps_.at(i).get_model();
                    const mat inv_Sigma_w = inv_sympd(promp.get_Sigma_w());
                    const mat inv_Sigma_y = inv_sympd(promp.get_Sigma_y());
                    const vec mu_w = promp.get_mu_w();

                    vector<double> mult_c;
                    vector<double> denominator_Sigma_y;
                    for(int t = min_duration - 1; t < nobs; t++) {
                        for(int d = 0; d < ndurations; d++) {
                            int first_idx_seg = t - min_duration - d + 1;
                            if (first_idx_seg < 0)
                                break;
                            mult_c.push_back(eta(i, d, t));
                            denominator_Sigma_y.push_back(eta(i, d, t) +
                                    log(min_duration + d));
                        }
                    }

                    // Computing the multiplicative constants for mu_w and
                    // Sigma_w.
                    vec mult_c_normalized(mult_c);
                    mult_c_normalized -= logsumexp(mult_c_normalized);
                    mult_c_normalized = exp(mult_c_normalized);
 
                    // Computing the multiplicative constants for Sigma_y since
                    // they have a different denominator.
                    vec mult_c_Sigma_y_normalized(mult_c);
                    vec den_Sigma_y(denominator_Sigma_y);
                    mult_c_Sigma_y_normalized -= logsumexp(den_Sigma_y);
                    mult_c_Sigma_y_normalized = exp(mult_c_Sigma_y_normalized);

                    mat new_Sigma_y(size(promp.get_Sigma_y()), fill::zeros);

                    // EM for ProMPs.
                    vec weighted_sum_post_mean(size(mu_w), fill::zeros);
                    mat weighted_sum_post_cov(size(inv_Sigma_w), fill::zeros);
                    mat weighted_sum_post_mean_mean_T(size(inv_Sigma_w),
                            fill::zeros);
                    int idx_mult_c = 0;
                    for(int t = min_duration - 1; t < nobs; t++) {
                        for(int d = 0; d < ndurations; d++) {
                            int first_idx_seg = t - min_duration - d + 1;
                            if (first_idx_seg < 0)
                                break;
                            const int current_duration = min_duration + d;
                            const cube& Phis = getPhiCube(i, current_duration);

                            // E step for the emission hidden variables (Ws).
                            // Computing the posterior of W given Y and Theta.

                            // Computing the posterior covariance of the hidden
                            // variable w for this segment.
                            mat posterior_cov(size(inv_Sigma_w), fill::zeros);
                            for(int step = 0; step < current_duration; step++) {
                                posterior_cov += Phis.slice(step).t() *
                                    inv_Sigma_y * Phis.slice(step);
                            }
                            posterior_cov = (posterior_cov+posterior_cov.t())/2.0;
                            posterior_cov = posterior_cov + inv_Sigma_w;
                            posterior_cov = (posterior_cov+posterior_cov.t())/2.0;
                            posterior_cov = inv_sympd(posterior_cov);
                            posterior_cov = (posterior_cov+posterior_cov.t())/2.0;

                            // Computing the posterior mean of the hidden
                            // variable w for this segment.
                            vec posterior_mean(size(mu_w), fill::zeros);
                            for(int step = 0; step < current_duration; step++) {
                                const vec& ob = obs.col(first_idx_seg + step);
                                posterior_mean += Phis.slice(step).t() *
                                    inv_Sigma_y * ob;
                            }
                            posterior_mean = inv_Sigma_w * mu_w + posterior_mean;
                            posterior_mean = posterior_cov * posterior_mean;

                            // Getting the multiplicative constants.
                            double mult_constant = mult_c_normalized(idx_mult_c);
                            double mult_constant_Sigma_y =
                                    mult_c_Sigma_y_normalized(idx_mult_c);
                            idx_mult_c++;

                            // Statistics required for updating mu_w & Sigma_w.
                            weighted_sum_post_mean += mult_constant *
                                    posterior_mean;
                            weighted_sum_post_cov += mult_constant *
                                    posterior_cov;
                            weighted_sum_post_mean_mean_T += mult_constant *
                                    posterior_mean * posterior_mean.t();

                            // Computing the new output noise covariance: Sigma_y.
                            mat Sigma_y_term(size(new_Sigma_y), fill::zeros);
                            for(int step = 0; step < current_duration; step++) {
                                const mat& phi = Phis.slice(step);
                                const vec& diff_y = obs.col(first_idx_seg + step) -
                                    phi * posterior_mean;
                                Sigma_y_term += diff_y * diff_y.t() +
                                    phi * posterior_cov * phi.t();
                            }
                            new_Sigma_y += mult_constant_Sigma_y * Sigma_y_term;
                        }
                    }

                    // M step for the emission variables.
                    vec new_mu_w(weighted_sum_post_mean);
                    mat new_Sigma_w_MLE = weighted_sum_post_cov +
                        weighted_sum_post_mean_mean_T - new_mu_w*new_mu_w.t();

                    // If there is a prior for Sigma_w then we do MAP instead.
                    mat new_Sigma_w;
                    if (Sigma_w_prior_) {
                        double v_0 = Sigma_w_prior_->getDof();
                        double D = mu_w.n_rows;
                        mat S_0 = Sigma_w_prior_->getPhi();
                        double mle_den = exp(logsumexp(mult_c));
                        cout << "State " << i << " MLE Den: " << mle_den <<
                                endl;
                        new_Sigma_w = (S_0 + mle_den * new_Sigma_w_MLE) /
                                (v_0 + mle_den + D + 2);
                    }
                    else
                        new_Sigma_w = new_Sigma_w_MLE;

                    // Checking that the new Sigma_w is a covariance matrix.
                    vec eigenvalues_map = eig_sym(new_Sigma_w);
                    assert(eigenvalues_map(0) > 0);

                    // Setting the new parameters.
                    promp.set_mu_w(new_mu_w);
                    promp.set_Sigma_w(new_Sigma_w);
                    promp.set_Sigma_y(new_Sigma_y);
                    promps_.at(i).set_model(promp);
                }
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

            nlohmann::json to_stream() const {
                vector<nlohmann::json> array_emission_params;
                for(int i = 0; i < getNumberStates(); i++) {
                    const ProMP& promp = promps_.at(i).get_model();
                    nlohmann::json whole_thing;
                    nlohmann::json promp_params;
                    promp_params["mu_w"] = vec2json(promp.get_mu_w());
                    promp_params["Sigma_w"] = mat2json(promp.get_Sigma_w());
                    promp_params["Sigma_y"] = mat2json(promp.get_Sigma_y());
                    whole_thing["model"] = promp_params;
                    whole_thing["num_joints"] = getDimension();

                    // Note that the information about the basis functions is not
                    // serialized.
                    array_emission_params.push_back(whole_thing);
                }
                nlohmann::json ret = array_emission_params;
                return ret;
            }

            void from_stream(const nlohmann::json& emission_params) {
                // Note that the given parameters need to be consistent with some
                // of the preexisting settings. Moreover, part of the structure of
                // this emission process is not read from the input json.
                // (e.g. the used basis functions).
                assert(emission_params.size() == getNumberStates());
                for(int i = 0; i < getNumberStates(); i++) {
                    const nlohmann::json& params = emission_params.at(i);
                    promps_.at(i).set_model(json2basic_promp(params.at("model")));
                    assert(params.at("num_joints") == getDimension());
                }
                return;
            }

            void set_Sigma_w_Prior(InverseWishart prior) {
                Sigma_w_prior_ = std::make_shared<InverseWishart>(
                        std::move(prior));
                int size_cov = promps_.at(0).get_model().get_Sigma_w().n_rows;
                assert(size_cov == Sigma_w_prior_->getPhi().n_rows);
            }


        private:

            // Returns the marginal distribution of a particular state (ProMP) and
            // duration. Keep in mind that the covariance matrix grows quadratically
            // with respect to the duration. This could be cached for efficiency
            // but is mostly intended for debugging.
            random::NormalDist getNormalDistForMultipleTimeSteps(int state, int duration) {
                int nrows = getDimension();
                const FullProMP& promp = promps_.at(state);
                const mat stacked_Phi = getPhiStacked(state, duration);
                vec mean = stacked_Phi * promp.get_model().get_mu_w();
                mat cov = stacked_Phi * promp.get_model().get_Sigma_w() *
                    stacked_Phi.t();

                // Adding the noise variance.
                mat noise_cov(size(cov), fill::zeros);
                for(int i = 0; i < duration; i++)
                    noise_cov.submat(i * nrows, i * nrows, (i + 1) * nrows - 1,
                            (i + 1) * nrows - 1) = promp.get_model().get_Sigma_y();

                cov = cov + noise_cov;
                return random::NormalDist(mean, cov);
            }

            // Takes the covariance matrix for omega and returns it as if the
            // joints were independent (blockdiag operator in Sebastian's paper).
            mat getCovarianceIndependentJoints(const mat& cov) const {
                mat ret(size(cov), fill::zeros);
                assert(cov.n_rows == cov.n_cols);
                int dim = cov.n_rows;
                assert(dim % getDimension() == 0);
                int blocklen = dim / getDimension();
                for(int i = 0; i < dim; i++)
                    for(int j = 0; j < dim; j++)
                        if ((i/blocklen) == (j/blocklen))
                            ret(i, j) = cov(i, j);
                mat extra_term = eye<mat>(size(ret)) * 1e-4;
                return ret + extra_term;
            }

            mat getPhiStacked(int state, int duration) {
                cube Phis = getPhiCube(state, duration);
                mat PhiStacked(Phis.n_rows * Phis.n_slices, Phis.n_cols);
                for(int d = 0; d < duration; d++)
                    PhiStacked.rows(d * Phis.n_rows, (d + 1) * Phis.n_rows - 1) =
                        Phis.slice(d);
                return PhiStacked;
            }

            cube getPhiCube(int state, int duration) {
                pair<int, int> p = make_pair(state, duration);
                if (cachePhis_.find(p) != cachePhis_.end())
                    return cachePhis_[p];
                const FullProMP& promp = promps_.at(state);

                // The samples are assumed to be equally spaced.
                vec sample_locations = linspace<vec>(0, 1.0, duration);
                mat tmp = promp.get_phi_t(0);
                int ncols = tmp.n_cols;
                int nrows = tmp.n_rows;
                cube stacked_Phi(nrows, ncols, duration, fill::zeros);
                for(int i = 0; i < duration; i++)
                    stacked_Phi.slice(i) = promp.get_phi_t(sample_locations(i));
                cachePhis_[p] = stacked_Phi;
                return stacked_Phi;
            }

            map<pair<int, int>, cube> cachePhis_;
            vector<FullProMP> promps_;
            std::shared_ptr<InverseWishart> Sigma_w_prior_;
    };

};

#endif
