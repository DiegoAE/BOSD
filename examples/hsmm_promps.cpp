#include <armadillo>
#include <ForwardBackward.hpp>
#include <HSMM.hpp>
#include <iostream>
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

        double loglikelihoodBatchVersion(int state, const arma::mat& obs) const {
            int dimension = obs.n_rows;
            assert(dimension == getDimension());
            random::NormalDist dist = getNormalDistForMultipleTimeSteps(state,
                    obs.n_cols);
            vec stacked_obs = vectorise(obs);
            return random::log_normal_density(dist, stacked_obs);
        }

        void reestimate(int min_duration, const arma::cube& eta,
                const arma::mat& obs) {
            // TODO
            return;
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

    private:

        // Returns the marginal distribution of a particular state (ProMP) and
        // duration. Keep in mind that the covariance matrix grows quadratically
        // with respect to the duration. This could be cached for efficiency
        // but is mostly intended for debugging.
        random::NormalDist getNormalDistForMultipleTimeSteps(int state, int duration) const {
            int nrows = getDimension();
            const FullProMP& promp = promps_.at(state);
            mat stacked_Phi = getStackedPhi(state, duration);
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

        mat getStackedPhi(int state, int duration) const {
            const FullProMP& promp = promps_.at(state);

            // The samples are assumed to be equally spaced.
            vec sample_locations = linspace<vec>(0, 1.0, duration);
            mat tmp = promp.get_phi_t(0);
            int ncols = tmp.n_cols;
            int nrows = tmp.n_rows;
            mat stacked_Phi(nrows * duration, ncols, fill::zeros);
            for(int i = 0; i < duration; i++) {

                // Stacking vertically multiple Phis for different time steps.
                stacked_Phi.rows(i * nrows, (i + 1) * nrows - 1) =
                        promp.get_phi_t(sample_locations(i));
            }
            return stacked_Phi;
        }

        vector<FullProMP> promps_;
};

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
        mat Sigma_w = 100*eye<mat>(n_basis_functions * njoints,
                    n_basis_functions * njoints);
        mat Sigma_y = 0.0001*eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(kernel, promp, njoints);
        promps.push_back(poly);
    }

    // Creating the ProMP emission.
    shared_ptr<AbstractEmission> ptr_emission(new ProMPsEmission(promps));

    HSMM promp_hsmm(ptr_emission, transition, pi, durations, min_duration);

    int nsegments = 10;
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

    return 0;
}