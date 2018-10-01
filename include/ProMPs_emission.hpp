#ifndef PROMP_EMISSION_H
#define PROMP_EMISSION_H

#include <armadillo>
#include <emissions.hpp>
#include <iostream>
#include <json.hpp>
#include <ForwardBackward.hpp>
#include <map>
#include <memory>
#include <robotics.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;

namespace hsmm {

    class NormalInverseWishart {
        public:
            NormalInverseWishart(vec mu_0, double lambda, mat Phi, int dof) :
                    Phi_(Phi), dof_(dof), mu_0_(mu_0), lambda_(lambda) {
                assert(dof > Phi.n_rows + 1);
                vec eigenvalues = eig_sym(Phi);
                assert(eigenvalues(0) > 0);
                assert(!(lambda < 0));
                assert(mu_0.n_rows == Phi.n_rows);
            }

            // This constructor assumes there is no prior for the mean.
            // (i.e. only the inverse-Wishart part).
            NormalInverseWishart(mat Phi, int dof) : NormalInverseWishart(
                    zeros<vec>(Phi.n_rows), 0.0, Phi, dof) {}

            vec getMu0() {
                return mu_0_;
            }

            double getLambda() {
                return lambda_;
            }

            mat getPhi() {
                return Phi_;
            }

            int getDof() {
                return dof_;
            }

        private:
            vec mu_0_;
            double lambda_;
            mat Phi_;
            int dof_;
    };

    double zeroMeanGaussianLogLikelihood(const vec& x, const mat &precision) {
        return -0.5 * as_scalar(x.t()*precision*x - log(det(precision)) +
                x.n_elem * log(2*datum::pi));
    }


    class ProMPsEmission : public AbstractEmissionOnlineSetting {
        public:
            ProMPsEmission(vector<FullProMP> promps) :
                    AbstractEmissionOnlineSetting(promps.size(),
                    promps.at(0).get_num_joints()), promps_(promps),
                    diagonal_sigma_y_(true) {
                for(int i = 0; i < getNumberStates(); i++)
                    assert(promps_.at(i).get_num_joints() == getDimension());
            }

            ProMPsEmission* clone() const {
                return new ProMPsEmission(*this);
            }

            void resetMemCaches() {
                cachePhis_.clear();
                cacheInvS_.clear();
                cacheK_.clear();
                cachePosteriorSigma_.clear();
            }

            // This initialization mechanism assumes all the hidden states
            // share the same basis functions and their hyperparameters.
            void init_params_from_data(int min_duration, int ndurations,
                    const arma::field<arma::field<arma::mat>>& mobs) {
                vector<pair<double, vec>> pairs;
                vector<double> noise_vars;
                for(auto &obs : mobs) {
                    for(int t = 0; t < obs.n_elem; t++) {
                        for(int d = 0; d < ndurations; d++) {
                            if (t + min_duration + d > obs.n_elem)
                                break;
                            int end_idx = t + min_duration + d - 1;
                            auto &segment = obs.rows(t, end_idx);
                            vec lsq_omega = least_squares_omega(0, segment);
                            double var = var_isotropic_gaussian_given_omega(0,
                                    lsq_omega, segment);
                            pairs.push_back(make_pair(var, lsq_omega));
                            noise_vars.push_back(var);
                        }
                    }
                }
                sort(pairs.begin(), pairs.end(), [](const pair<double, vec>& a,
                        const pair<double, vec>& b)
                        {return a.first < b.first;});
                int cutoff_index = (int)((pairs.size() - 1) *
                        init_fraction_);
                double threshold = pairs.at(cutoff_index).first;
                cout << "Cuttoff index for init: " << cutoff_index <<
                        " var: " << threshold << endl;

                // Showing a histogram of the distribution of noise vars.
                vec vars(noise_vars);
                vec histogram_cutoffs = linspace<vec>(0, max(vars), 20);
                uvec histogram = histc(vars, histogram_cutoffs);
                cout << "Hist. of noise vars:" << endl << histogram << endl;

                mat remaining_w(promps_.at(0).get_model().get_mu_w().n_rows,
                        cutoff_index + 1);
                for(int i = 0; i <= cutoff_index; i++)
                    remaining_w.col(i) = pairs.at(i).second;
                mat means;
                kmeans(means, remaining_w, getNumberStates(), static_subset,
                        10, false);

                // Updating the means of the ProMPs based on the k-means.
                for(int i = 0; i < getNumberStates(); i++) {
                    ProMP promp = promps_.at(i).get_model();
                    promp.set_mu_w(means.col(i));
                    promps_.at(i).set_model(promp);
                }
            }

            // Note that the method is returning the posterior covariance for
            // the given state and duration which is independent of obs.
            mat generateCachedMatrices(const pair<int, int> &p) const {
                if (cacheInvS_.find(p) == cacheInvS_.end() ||
                        cacheK_.find(p) == cacheK_.end()) {
                    int state = p.first;
                    int dur = p.second;
                    const FullProMP& promp = promps_.at(state);
                    const cube& Phis = getPhiCube(state, dur);
                    mat Sigma(promp.get_model().get_Sigma_w());
                    mat Sigma_y(promp.get_model().get_Sigma_y());
                    field<mat> invS(dur);
                    field<mat> K(dur);
                    for(int i = 0; i < dur; i++) {
                        const mat& Phi = Phis.slice(i);
                        mat S = Phi * Sigma * Phi.t() + Sigma_y;
                        invS(i) = inv_sympd(S);

                        // Kalman updating (correcting) step.
                        K(i) = Sigma * Phi.t() * inv(S);
                        Sigma = Sigma - K(i) * S * K(i).t();
                    }

                    // Caching the matrices for faster likelihood evaluation.
                    cacheInvS_[p] = invS;
                    cacheK_[p] = K;
                    cachePosteriorSigma_[p] = Sigma;
                }
                return cachePosteriorSigma_.at(p);
            }

            double loglikelihood(int state, const field<mat>& obs) const {

                // Making sure all the required matrices are already
                // precomputed.
                pair<int, int> p = make_pair(state, obs.n_elem);
                generateCachedMatrices(p);

                const FullProMP& promp = promps_.at(state);
                const cube& Phis = getPhiCube(state, obs.n_elem);
                vec mu(promp.get_model().get_mu_w());
                const field<mat>& invS = cacheInvS_[p];
                const field<mat>& K = cacheK_[p];
                double ret = 0;
                for(int i = 0; i < obs.n_elem; i++) {
                    const mat& Phi = Phis.slice(i);
                    if (obs(i).is_empty()) {

                        // Making sure all the missing obs are at the end.
                        // Other missing obs patterns are not supported yet.
                        for(int j = i; j < obs.n_elem; j++)
                            assert(obs(j).is_empty());
                        break;
                    }
                    vec diff = obs(i) - Phi * mu;

                    // p(y_t | y_1, ..., y_{t-1}).
                    ret += zeroMeanGaussianLogLikelihood(diff, invS(i));
                    mu = mu + K(i) * diff;
                }
                return ret;
            }

            double informationFilterLoglikelihood(int state,
                    const field<mat>& obs) const {
                const FullProMP& promp = promps_.at(state);

                // The samples are assumed to be equally spaced.
                vec sample_locations = linspace<vec>(0, 1.0, obs.n_elem);

                // Moment based parameterization.
                vec mu(promp.get_model().get_mu_w());
                mat Sigma(promp.get_model().get_Sigma_w());
                mat Sigma_y(promp.get_model().get_Sigma_y());

                // Canonical parameterization.
                mat information_matrix = inv_sympd(Sigma);
                vec information_state = information_matrix * mu;
                mat inv_obs_noise = inv_sympd(Sigma_y);

                double ret = 0;
                for(int i = 0; i < obs.n_elem; i++) {
                    mat Phi = promp.get_phi_t(sample_locations(i));

                    // Computing the likelihood under the current filtering
                    // distribution.
                    mat filtered_Sigma = inv_sympd(information_matrix);
                    vec filtered_mu = filtered_Sigma * information_state;

                    // p(y_t | y_{1:t-1}).
                    random::NormalDist dist = random::NormalDist(
                            Phi * filtered_mu,
                            Phi * filtered_Sigma * Phi.t() + Sigma_y);
                    ret = ret + log_normal_density(dist, obs(i));

                    mat aux = Phi.t() * inv_obs_noise;
                    mat I_k = aux * Phi;
                    mat i_k = aux * obs(i);
                    information_matrix = information_matrix + I_k;
                    information_state = information_state + i_k;
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

            void reestimate(int min_duration,
                    const arma::field<arma::cube>& meta,
                    const arma::field<arma::field<arma::mat>>& mobs) {
                int nseq = mobs.n_elem;
                for(int i = 0; i < getNumberStates(); i++) {
                    ProMP promp = promps_.at(i).get_model();
                    const mat inv_Sigma_w = inv_sympd(promp.get_Sigma_w());
                    const mat inv_Sigma_y = inv_sympd(promp.get_Sigma_y());
                    const vec mu_w = promp.get_mu_w();

                    vector<double> mult_c;
                    vector<double> denominator_Sigma_y;
                    for(int s = 0; s < nseq; s++) {
                        const cube& eta = meta(s);
                        int nobs = mobs(s).n_elem;
                        int ndurations = eta.n_cols;
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
                    for(int s = 0; s < nseq; s++) {
                        auto& obs = mobs(s);
                        int nobs = obs.n_elem;
                        int ndurations = meta(s).n_cols;
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
                                    const vec& ob = obs(first_idx_seg + step);
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
                                    const vec& diff_y = obs(first_idx_seg + step) -
                                        phi * posterior_mean;
                                    Sigma_y_term += diff_y * diff_y.t() +
                                        phi * posterior_cov * phi.t();
                                }
                                new_Sigma_y += mult_constant_Sigma_y * Sigma_y_term;
                            }
                        }
                    }

                    // Expected number of segments generated by the i-th state.
                    double mle_den = exp(logsumexp(mult_c));

                    // M step for the emission variables.
                    vec new_mu_w_MLE(weighted_sum_post_mean);
                    mat new_Sigma_w_MLE = weighted_sum_post_cov +
                        weighted_sum_post_mean_mean_T - new_mu_w_MLE *
                        new_mu_w_MLE.t();

                    // If there is a prior then we do MAP instead.
                    mat new_Sigma_w;
                    mat new_mu_w;
                    if (normal_inverse_prior_) {
                        double v_0 = normal_inverse_prior_->getDof();
                        double D = mu_w.n_rows;
                        mat S_0 = normal_inverse_prior_->getPhi();
                        new_Sigma_w = (S_0 + mle_den * new_Sigma_w_MLE) /
                                (v_0 + mle_den + D + 2);

                        double k_0 = normal_inverse_prior_->getLambda();
                        vec m_0 = normal_inverse_prior_->getMu0();
                        new_mu_w = (k_0 * m_0 + mle_den * new_mu_w_MLE) /
                                (mle_den + k_0);
                    }
                    else {
                        new_Sigma_w = new_Sigma_w_MLE;
                        new_mu_w = new_mu_w_MLE;
                    }

                    cout << "State " << i << " MLE Den: " << mle_den << " ";
                    if (mle_den > epsilon_) {

                        // Making sure the noise covariance is diagonal.
                        if (diagonal_sigma_y_)
                            new_Sigma_y = diagmat(new_Sigma_y.diag());

                        // Checking that the new Sigma_w is a covariance matrix.
                        vec eigenvalues_map = eig_sym(new_Sigma_w);
                        assert(eigenvalues_map(0) > 0);

                        // Setting the new parameters.
                        promp.set_mu_w(new_mu_w);
                        promp.set_Sigma_w(new_Sigma_w);
                        promp.set_Sigma_y(new_Sigma_y);
                        promps_.at(i).set_model(promp);
                        resetMemCaches();
                        cout << ". Updated." << endl;
                    }
                    else
                        cout << ". Not updated." << endl;
                }
            }

            // Unshadowing this method from the AbstractEmission class.
            using AbstractEmission::sampleFromState;

            field<mat> sampleFromState(int state, int size,
                    mt19937 &rand_generator) const {
                return sampleFromProMP(promps_.at(state), size, rand_generator);
            }

            field<mat> sampleFromProMP(const FullProMP& fpromp, int size,
                    mt19937 &rand_generator) const {
                const ProMP& model = fpromp.get_model();
                vector<vec> w_samples = random::sample_multivariate_normal(
                        rand_generator,
                        {model.get_mu_w(), model.get_Sigma_w()}, 1);
                vec w = w_samples.back();

                vec noise_mean = zeros<vec>(getDimension());
                vector<vec> output_noise = random::sample_multivariate_normal(
                        rand_generator, {noise_mean, model.get_Sigma_y()}, size);

                field<mat> ret(size);

                // The samples are assumed to be equally spaced.
                vec sample_locations = linspace<vec>(0, 1.0, size);
                for(int i = 0; i < size; i++) {
                    double z = sample_locations(i);
                    mat phi_z =fpromp.get_phi_t(z);
                    ret(i) = phi_z * w + output_noise.at(i);
                }
                return ret;
            }

            // Equivalent to sampleFromState but uses conditioning.
            field<mat> sampleFromState2(int state, int size,
                    mt19937 &rand_generator) const {
                field<mat> no_obs;
                return sampleNextObsGivenPastObs(state, size, no_obs,
                        rand_generator);
            }

            field<mat> sampleNextObsGivenPastObs(int state, int seg_dur,
                    const field<mat>& past_obs, std::mt19937 &rng) const {
                assert(past_obs.n_elem < seg_dur);

                // Making sure all the required matrices are already computed.
                pair<int, int> p = make_pair(state, seg_dur);
                generateCachedMatrices(p);

                field<mat> ret(seg_dur);
                for(int i = 0; i < past_obs.n_elem; i++)
                    ret(i) = past_obs(i);

                const FullProMP& promp = promps_.at(state);
                const cube& Phis = getPhiCube(state, seg_dur);
                vec mu(promp.get_model().get_mu_w());

                const field<mat>& invS = cacheInvS_.at(p);
                const field<mat>& K = cacheK_.at(p);
                int i;
                for(i = 0; i < past_obs.n_elem; i++)
                    mu = mu + K(i) * (past_obs(i) - Phis.slice(i) * mu);

                // Now i indexes the offset we want to sample from.
                for(; i < seg_dur; i++) {
                    vector<vec> sample = random::sample_multivariate_normal(
                            rng, {Phis.slice(i) * mu, inv(invS(i))}, 1);
                    ret(i) = sample.at(0);
                    mu = mu + K(i) * (ret(i) - Phis.slice(i) * mu);
                }
                return ret;
            }

            field<mat> sampleFirstSegmentObsGivenLastSegment(int curr_state,
                    int curr_seg_dur, const field<mat> &last_segment,
                    int last_state, std::mt19937 &rng) const {
                mat last_obs = last_segment(last_segment.n_elem - 1);

                // Making sure all the required matrices are already computed.
                pair<int, int> p = make_pair(curr_state, curr_seg_dur);
                pair<int, int> last_p = make_pair(last_state,
                        last_segment.n_elem);
                generateCachedMatrices(p);
                mat last_p_Sigma = generateCachedMatrices(last_p);

                // Finding the posterior omega mean for the last segment.
                const cube& LPhis = getPhiCube(last_state, last_segment.n_elem);
                vec last_p_mu(promps_.at(last_state).get_model().get_mu_w());
                const field<mat>& last_K = cacheK_[last_p];
                for(int i = 0; i < last_segment.n_elem; i++) {
                    const mat& Phi = LPhis.slice(i);
                    vec diff = last_segment(i) - Phi * last_p_mu;
                    last_p_mu = last_p_mu + last_K(i) * diff;
                }
                FullProMP last_full_promp(promps_.at(last_state));
                mat zeros_Sigma_y = zeros<mat>(getDimension(), getDimension());
                ProMP last_p_promp(last_p_mu, last_p_Sigma, zeros_Sigma_y);
                last_full_promp.set_model(last_p_promp);

                // This gives the posterior distribution over q = (y, v) at the
                // last time step of the last segment.
                random::NormalDist last_pos_vel = last_full_promp.joint_dist(
                        1.0, true, true, false);
                vec last_p_pos = last_pos_vel.mean().head(getDimension());
                vec last_p_vel = last_pos_vel.mean().tail(getDimension());

                FullProMP promp = promps_.at(curr_state);
                promp = promp.condition_current_position(0, 1.0, last_p_pos);
                //promp = promp.condition_current_state(0, 1.0, last_p_pos,
                //        last_p_vel);
                return sampleFromProMP(promp, curr_seg_dur, rng);
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
                resetMemCaches();
                return;
            }

            // TODO: change the name of this method.
            void set_Sigma_w_Prior(NormalInverseWishart prior) {
                normal_inverse_prior_ = std::make_shared<NormalInverseWishart>(
                        std::move(prior));
                int size_cov = promps_.at(0).get_model().get_Sigma_w().n_rows;
                assert(size_cov == normal_inverse_prior_->getPhi().n_rows);
            }

            void setParamsForInitialization(double fraction) {
                assert(fraction > 0 && fraction < 1.0);
                init_fraction_ = fraction;
            }


        protected:

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

            cube getPhiCube(int state, int duration) const {
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

            // Returns the least squares solution to the problem:
            // y(t) = Phi(t) * w for a bunch of i.i.d. y's. Note that omega is
            // not assumed to be a random variable.
            vec least_squares_omega(int state,
                    const field<mat>& segment) const {
                int nobs = segment.n_elem;
                const cube& Phis = getPhiCube(state, nobs);
                mat A(size(Phis.slice(0).t() * Phis.slice(0)),
                        fill::zeros);
                vec b(A.n_rows, fill::zeros);
                for(int t = 0; t < nobs; t++) {
                    const mat& Phi = Phis.slice(t);
                    A = A + Phi.t() * Phi;
                    b = b + Phi.t() * segment(t);
                }

                // Solving A * w = b
                vec lsq_omega = solve(A, b);
                return lsq_omega;
            }

            // Compute the variance of an isotropic Gaussian noise model
            // given omega.
            double var_isotropic_gaussian_given_omega(int state, vec w,
                    const field<mat>& segment) const {
                int nobs = segment.n_elem;
                const cube& Phis = getPhiCube(state, nobs);
                double var = 0;
                for(int t = 0; t < nobs; t++) {
                    const mat& Phi = Phis.slice(t);
                    vec diff = segment(t) - Phi * w;
                    var += dot(diff, diff);
                }
                return var / (nobs * segment(0).n_rows);
            }


            vector<FullProMP> promps_;
            std::shared_ptr<NormalInverseWishart> normal_inverse_prior_;
            double epsilon_ = 1e-15;
            bool diagonal_sigma_y_;

            // Fraction of the total least squares omega estimates that will be
            // used for initialization.
            double init_fraction_ = 0.1;

            // Members for caching.
            mutable map<pair<int, int>, cube> cachePhis_;
            mutable map<pair<int, int>, field<mat>> cacheInvS_;
            mutable map<pair<int, int>, field<mat>> cacheK_;
            mutable map<pair<int, int>, mat> cachePosteriorSigma_;
    };


    // This version of the ProMP is specifically designed for HMMs where a
    // single observation is a time series itself
    class ProMPsEmissionHMM : public ProMPsEmission {
        public:
            ProMPsEmissionHMM(vector<FullProMP> promps) :
                    ProMPsEmission(promps) {}

            ProMPsEmission* clone() const {
                return new ProMPsEmission(*this);
            }

            double loglikelihood(int state, const field<mat>& obs) const {

                // Making sure that the duration is one.
                assert(obs.n_elem == 1);
                auto& sequence = obs(0);
                const FullProMP& promp = promps_.at(state);

                // The samples are assumed to be equally spaced.
                vec sample_locations = linspace<vec>(0, 1.0, sequence.n_cols);

                vec mu(promp.get_model().get_mu_w());
                mat Sigma(promp.get_model().get_Sigma_w());
                mat Sigma_y(promp.get_model().get_Sigma_y());
                double ret = 0;
                for(int i = 0; i < sequence.n_cols; i++) {
                    mat Phi = promp.get_phi_t(sample_locations(i));
                    mat S = Phi * Sigma * Phi.t() + Sigma_y;

                    // Required for the marginal likelihood p(y_t | y_{1:t-1}).
                    random::NormalDist dist = random::NormalDist(Phi * mu, S);
                    ret = ret + log_normal_density(dist, sequence.col(i));

                    // Using the kalman updating step to compute this efficiently.
                    mat K = Sigma * Phi.t() * inv(S);
                    mu = mu + K * (sequence.col(i) - Phi * mu);
                    Sigma = Sigma - K * S * K.t();
                }
                return ret;
            }

            void reestimate(int min_duration,
                    const arma::field<arma::cube>& meta,
                    const arma::field<arma::field<arma::mat>>& mobs) {
                int nseq = mobs.n_elem;
                for(int i = 0; i < getNumberStates(); i++) {
                    ProMP promp = promps_.at(i).get_model();
                    const mat inv_Sigma_w = inv_sympd(promp.get_Sigma_w());
                    const mat inv_Sigma_y = inv_sympd(promp.get_Sigma_y());
                    const vec mu_w = promp.get_mu_w();

                    vector<double> mult_c;
                    vector<double> denominator_Sigma_y;
                    for(int s = 0; s < nseq; s++) {
                        const cube& eta = meta(s);
                        int nobs = mobs(s).n_elem;
                        int ndurations = eta.n_cols;
                        assert(ndurations == 1 && min_duration == 1);
                        for(int t = 0; t < nobs; t++) {
                            mult_c.push_back(eta(i, 0, t));
                            int current_duration = mobs(s)(t).n_cols;
                            denominator_Sigma_y.push_back(eta(i, 0, t) +
                                    log(current_duration));
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
                    for(int s = 0; s < nseq; s++) {
                        auto& obs = mobs(s);
                        int nobs = obs.n_elem;
                        int ndurations = meta(s).n_cols;
                        for(int t = 0; t < nobs; t++) {

                                // Length of the current observation
                                const int current_duration = obs(t).n_cols;
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
                                    const vec& ob = obs(t).col(step);
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
                                    const vec& diff_y = obs(t).col(step) -
                                        phi * posterior_mean;
                                    Sigma_y_term += diff_y * diff_y.t() +
                                        phi * posterior_cov * phi.t();
                                }
                                new_Sigma_y += mult_constant_Sigma_y * Sigma_y_term;
                        }
                    }

                    // Expected number of segments generated by the i-th state.
                    double mle_den = exp(logsumexp(mult_c));

                    // M step for the emission variables.
                    vec new_mu_w_MLE(weighted_sum_post_mean);
                    mat new_Sigma_w_MLE = weighted_sum_post_cov +
                        weighted_sum_post_mean_mean_T - new_mu_w_MLE *
                        new_mu_w_MLE.t();

                    // If there is a prior then we do MAP instead.
                    mat new_Sigma_w;
                    mat new_mu_w;
                    if (normal_inverse_prior_) {
                        double v_0 = normal_inverse_prior_->getDof();
                        double D = mu_w.n_rows;
                        mat S_0 = normal_inverse_prior_->getPhi();
                        new_Sigma_w = (S_0 + mle_den * new_Sigma_w_MLE) /
                                (v_0 + mle_den + D + 2);

                        double k_0 = normal_inverse_prior_->getLambda();
                        vec m_0 = normal_inverse_prior_->getMu0();
                        new_mu_w = (k_0 * m_0 + mle_den * new_mu_w_MLE) /
                                (mle_den + k_0);
                    }
                    else {
                        new_Sigma_w = new_Sigma_w_MLE;
                        new_mu_w = new_mu_w_MLE;
                    }

                    cout << "State " << i << " MLE Den: " << mle_den << " ";
                    if (mle_den > epsilon_) {

                        // Making sure the noise covariance is diagonal.
                        if (diagonal_sigma_y_)
                            new_Sigma_y = diagmat(new_Sigma_y.diag());

                        // Checking that the new Sigma_w is a covariance matrix.
                        vec eigenvalues_map = eig_sym(new_Sigma_w);
                        assert(eigenvalues_map(0) > 0);

                        // Setting the new parameters.
                        promp.set_mu_w(new_mu_w);
                        promp.set_Sigma_w(new_Sigma_w);
                        promp.set_Sigma_y(new_Sigma_y);
                        promps_.at(i).set_model(promp);
                        cout << ". Updated." << endl;
                    }
                    else
                        cout << ". Not updated." << endl;
                }
            }

            field<mat> sampleFromState(int state, int size,
                    mt19937 &rand_generator) const {

                // Since the segments are modeled as single observations.
                assert(size == 1);
                const ProMP& model = promps_.at(state).get_model();
                vector<vec> w_samples = random::sample_multivariate_normal(
                        rand_generator, {model.get_mu_w(), model.get_Sigma_w()}, 1);
                vec w = w_samples.back();

                // Getting the actual size of the segment.
                size = getDurationForEachSegment(rand_generator);

                vec noise_mean = zeros<vec>(getDimension());
                vector<vec> output_noise = random::sample_multivariate_normal(
                        rand_generator, {noise_mean, model.get_Sigma_y()}, size);

                mat joint_sample(getDimension(), size);

                // The samples are assumed to be equally spaced.
                vec sample_locations = linspace<vec>(0, 1.0, size);
                for(int i = 0; i < size; i++) {
                    double z = sample_locations(i);
                    mat phi_z = promps_.at(state).get_phi_t(z);
                    joint_sample.col(i) = phi_z * w + output_noise.at(i);
                }
                field<mat> ret = {joint_sample};
                return ret;
            }

        private:

            int getDurationForEachSegment(mt19937 &rand_generator) const {
                return 20;
            }

    };

};

#endif
