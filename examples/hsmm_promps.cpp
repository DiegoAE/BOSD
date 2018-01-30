#include <armadillo>
#include <ForwardBackward.hpp>
#include <HSMM.hpp>
#include <iostream>
#include <map>
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
            for(int i = 0; i < getNumberStates(); i++)
                assert(promps_.at(i).get_num_joints() == getDimension());
        }

        ProMPsEmission* clone() const {
            return new ProMPsEmission(*this);
        }

        double loglikelihood(int state, const arma::mat& obs) const {
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
                vector<double> mult_c;
                for(int t = min_duration - 1; t < nobs; t++) {
                    for(int d = 0; d < ndurations; d++) {
                        int first_idx_seg = t - min_duration - d + 1;
                        if (first_idx_seg < 0)
                            break;
                        mult_c.push_back(eta(i, d, t));
                    }
                }
                vec mult_c_normalized(mult_c);
                mult_c_normalized -= logsumexp(mult_c_normalized);
                mult_c_normalized = exp(mult_c_normalized);

                ProMP promp = promps_.at(i).get_model();
                const mat inv_Sigma_w = inv_sympd(promp.get_Sigma_w());
                const mat inv_Sigma_y = inv_sympd(promp.get_Sigma_y());
                const vec mu_w = promp.get_mu_w();

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
                            int obs_idx = first_idx_seg + step;
                            posterior_mean += Phis.slice(step).t() *
                                    inv_Sigma_y * obs.col(obs_idx);
                        }
                        posterior_mean = inv_Sigma_w * mu_w + posterior_mean;
                        posterior_mean = posterior_cov * posterior_mean;

                        double mult_constant = mult_c_normalized(idx_mult_c++);
                        weighted_sum_post_mean += mult_constant * posterior_mean;
                        weighted_sum_post_cov += mult_constant  * posterior_cov;
                        weighted_sum_post_mean_mean_T += mult_constant *
                                posterior_mean * posterior_mean.t();
                        // TODO: Sigma_y.
                    }
                }

                // M step for the emission variables.
                vec new_mu_w(weighted_sum_post_mean);

                mat term = new_mu_w * weighted_sum_post_mean.t();
                mat new_Sigma_w = weighted_sum_post_cov +
                        weighted_sum_post_mean_mean_T - term - term.t() +
                        new_mu_w * new_mu_w.t();

                // TODO.
                mat new_Sigma_y(size(inv_Sigma_y), fill::zeros);

                // Setting the new parameters.
                promp.set_mu_w(new_mu_w);
                promp.set_Sigma_w(new_Sigma_w);
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

        void printParameters() const {
            cout << "Means:" << endl;
            for(int i = 0; i < getNumberStates(); i++) {
                cout << "State " << i << ":" << endl << "Mean:" << endl <<
                        promps_.at(i).get_model().get_mu_w() << endl << "Cov"
                        << endl << promps_.at(i).get_model().get_Sigma_w() <<
                        endl;
            }
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
};


void PrintBestWeCanAimFor(int nstates, int ndurations, int min_duration,
        ivec hiddenStates, ivec hiddenDurations) {
    cout << "Best transition matrix we can aim at:" << endl;
    mat prueba(nstates, nstates, fill::zeros);
    for(int i = 0; i < hiddenStates.n_elem - 1; i++)
        prueba(hiddenStates(i), hiddenStates(i + 1))++;
    mat pruebasum = sum(prueba, 1);
    for(int i = 0; i < nstates; i++)
        prueba.row(i) /= pruebasum(i);
    cout << prueba << endl;

    cout << "Best duration matrix we can aim at:" << endl;
    mat emp_durations(nstates, ndurations, fill::zeros);
    for(int i = 0; i < hiddenStates.n_elem; i++)
        emp_durations(hiddenStates(i), hiddenDurations(i) - min_duration)++;
    mat emp_durations_sum = sum(emp_durations, 1);
    for(int i = 0; i < nstates; i++)
        emp_durations.row(i) /= emp_durations_sum(i);
    cout << emp_durations << endl;
}

void reset(HSMM& hsmm, vector<FullProMP> promps) {
    int nstates = hsmm.nstates_;
    int ndurations = hsmm.ndurations_;
    mat transition(hsmm.transition_);
    transition.fill(1.0/(nstates-1));
    transition.diag().zeros();  // No self-loops.
    hsmm.setTransition(transition);
    vec pi(hsmm.pi_);
    pi.fill(1.0/nstates);
    hsmm.setPi(pi);
    mat durations(hsmm.duration_);
    durations.fill(1.0/ndurations);
    hsmm.setDuration(durations);

    // Resetting emission.
    for(int i = 0; i < nstates; i++) {
        ProMP new_model = promps[i].get_model();
        vec new_mean = randn(size(new_model.get_mu_w()));
        mat new_Sigma_w(size(new_model.get_Sigma_w()), fill::eye);
        new_Sigma_w *= 100;
        new_model.set_mu_w(new_mean);
        new_model.set_Sigma_w(new_Sigma_w);
        promps[i].set_model(new_model);
    }
    shared_ptr<AbstractEmission> ptr_emission(new ProMPsEmission(promps));
    hsmm.setEmission(ptr_emission);
}

int main() {
    int ndurations = 4;
    int min_duration = 10;
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
        mu_w.fill(i * 10);
        mat Sigma_w = (i + 1)* 5 * eye<mat>(n_basis_functions * njoints,
                    n_basis_functions * njoints);
        mat Sigma_y = 0.0001*eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(kernel, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission.
    shared_ptr<AbstractEmission> ptr_emission(new ProMPsEmission(promps));

    HSMM promp_hsmm(ptr_emission, transition, pi, durations, min_duration);

    int nsegments = 100;
    ivec hidden_states, hidden_durations;
    mat toy_obs = promp_hsmm.sampleSegments(nsegments, hidden_states,
            hidden_durations);
    cout << "Generated states and durations" << endl;
    cout << join_horiz(hidden_states, hidden_durations) << endl;

    // Running the Viterbi algorithm.
    imat psi_duration(nstates, toy_obs.n_cols, fill::zeros);
    imat psi_state(nstates, toy_obs.n_cols, fill::zeros);
    mat delta(nstates, toy_obs.n_cols, fill::zeros);
    cube pdf = promp_hsmm.computeEmissionsLikelihood(toy_obs);
    Viterbi(transition, pi, durations, pdf, delta, psi_duration, psi_state,
            min_duration, toy_obs.n_cols);
    ivec viterbiStates, viterbiDurations;
    viterbiPath(psi_duration, psi_state, delta, viterbiStates,
            viterbiDurations);

    cout << "Viterbi states and durations" << endl;
    cout << join_horiz(viterbiStates, viterbiDurations) << endl;

    PrintBestWeCanAimFor(nstates, ndurations, min_duration, hidden_states,
            hidden_durations);

    // Learning the model from data.
    reset(promp_hsmm, promps);
    promp_hsmm.emission_->printParameters();
    promp_hsmm.fit(toy_obs, 100, 1e-10);
    promp_hsmm.emission_->printParameters();
    return 0;
}