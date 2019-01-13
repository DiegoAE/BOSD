#include <armadillo>
#include <cmath>
#include <emissions.hpp>
#include <HSMM.hpp>
#include <iostream>
#include <vector>

using namespace arma;
using namespace hsmm;
using namespace std;

#define NDIM 2
#define NDUR 70
#define MINDUR 30
#define NOISE_STDDEV 0.05

double gaussianlogpdf_(double x, double mu, double sigma) {
    double ret = ((x - mu)*(x - mu)) / (-2*sigma*sigma);
    ret = ret - log(sqrt(2 * M_PI) * sigma);
    return ret;
}

mat fieldToMat(int njoints, field<mat> &samples) {
    mat ret(njoints, samples.n_elem);
    for(int i = 0; i < samples.n_elem; i++)
        ret.col(i) = samples(i);
    return ret;
}

void fillDurationMatrix(mat& duration, const ivec& centers) {
    assert(centers.n_elem == duration.n_rows);
    duration.zeros();
    vec pmf_pattern = {0.1, 0.2, 0.4, 0.2, 0.1};
    for(int i = 0; i < centers.n_elem; i++) {
        int m = centers(i);
        int half = pmf_pattern.n_elem / 2;
        for(int j = 0; j < pmf_pattern.n_elem; j++)
            duration(i, m - half + j) = pmf_pattern(j);
    }
}

class ToyEmission : public AbstractEmissionOnlineSetting {
    public:
        ToyEmission(vector<function<vec(double)>> states, int ndim,
                double noise_stddev) : states_(states),
                AbstractEmissionOnlineSetting(states.size(), ndim) {
            output_gaussian_stddev_ = ones<mat>(getDimension(),
                    getNumberStates()) * noise_stddev;
        }

        ToyEmission* clone() const {
            return new ToyEmission(*this);
        }

        vec getSampleLocations(int length) const {
            return linspace<vec>(0, 1.0, length);
        }

        double loglikelihood(int state, const field<mat>& obs) const {
            vec t = getSampleLocations(obs.n_elem);
            double ret = 0;
            for(int i = 0; i < obs.n_elem; i++) {
                // Making sure all the missing obs are at the end.
                // Other missing obs patterns are not supported yet.
                if (obs(i).is_empty()) {
                    for(int j = i; j < obs.n_elem; j++)
                        assert(obs(j).is_empty());
                    break;
                }
                vec diff = obs(i) - states_.at(state)(t(i));
                for(int j = 0; j < diff.n_elem; j++)
                    ret += gaussianlogpdf_(diff(j), 0,
                            output_gaussian_stddev_(j, state));
            }
            return ret;
        }

        void reestimate(int min_duration, const field<cube>& meta,
                const field<field<mat>>& mobs ) {
            throw logic_error("Not implemented yet");
        }

        field<mat> sampleFromState(int state, int nsegments,
                mt19937& rng) const {
            vec t = getSampleLocations(nsegments);
            field<mat> ret(nsegments);
            for(int i = 0; i < nsegments; i++) {
                ret(i) = randn<vec>(getDimension());
                ret(i) = ret(i) % output_gaussian_stddev_.col(state);
                ret(i) = ret(i) + states_.at(state)(t(i));
            }
            return ret;
        }

        field<mat> sampleNextObsGivenPastObs(int state, int seg_dur,
                const field<mat>& past_obs, mt19937 &rng) const {
            throw logic_error("Not implemented yet");
        }

    private:
        vector<function<vec(double)>> states_;
        mat output_gaussian_stddev_;
};

vec halfsine1(double t) {
    assert(t <= 1.0 && t >= 0);
    t = t * M_PI;
    vec ret = {sin(t), -sin(t)};
    return ret;
}

vec halfsine2(double t) {
    assert(t <= 1.0 && t >= 0);
    t = t * M_PI;
    vec ret = {-sin(t), sin(t)};
    return ret;
}

vec halfsine3(double t) {
    assert(t <= 1.0 && t >= 0);
    t = t * M_PI;
    vec ret = {0.5*sin(t), 0.25*sin(t)};
    return ret;
}

vec halfsine4(double t) {
    assert(t <= 1.0 && t >= 0);
    t = t * M_PI;
    vec ret = {-0.25*sin(t), -0.5*sin(t)};
    return ret;
}

int main(int arc, char* argv[]) {
    vector<function<vec(double)>> states = {halfsine1, halfsine2,
            halfsine3, halfsine4};
    shared_ptr<ToyEmission> ptr_emission(new ToyEmission(states, NDIM,
            NOISE_STDDEV));
    vec pi(states.size());
    mat transition(states.size(), states.size());
    mat duration(states.size(), NDUR, fill::zeros);

    // Init the parameters.
    pi.fill(1.0 / states.size());
    transition.fill(1.0 / (states.size() - 1));
    transition.diag().zeros();
    ivec duration_centers = {5, 25, 45, 65};
    fillDurationMatrix(duration, duration_centers);

    OnlineHSMM online_toy_model(static_pointer_cast<
            AbstractEmissionOnlineSetting>(ptr_emission), transition, pi,
            duration, MINDUR);
    int nsegments = 12;
    ivec hs, hd;
    field<mat> toy_seq = online_toy_model.sampleSegments(nsegments, hs, hd);
    int nobs = toy_seq.n_elem;
    imat vit_mat = join_horiz(hs, hd);
    cout << "vit file" << endl << vit_mat << endl;

    // Online inference.
    mat state_marginals(states.size(), nobs);
    mat runlength_marginals(MINDUR + NDUR, nobs);
    mat implicit_remaining_marginals(MINDUR + NDUR, nobs);
    mat duration_marginals(NDUR, nobs);
    mat implicit_duration_marginals(NDUR, nobs);
    for(int c = 0; c < nobs; c++) {
        cout << "time step: " << c << endl;
        online_toy_model.addNewObservation(toy_seq(c));
        state_marginals.col(c) = online_toy_model.getStateMarginal();
        runlength_marginals.col(c) = online_toy_model.getRunlengthMarginal();
        duration_marginals.col(c) = online_toy_model.getDurationMarginal();
        implicit_remaining_marginals.col(c) = online_toy_model.
                getImplicitResidualTimeMarginal();
        //implicit_duration_marginals.col(c) = online_toy_model.
        //        getImplicitDurationMarginal();
    }

    // Saving everything.
    string prefix = "/local_data/dagudelo/paper_synth_experiment/";
    mat output = fieldToMat(NDIM, toy_seq);
    output.save(prefix + "paper_synth_exp.txt", raw_ascii);
    vit_mat.save(prefix + "paper_synth_gt_vit_seq.txt", raw_ascii);
    state_marginals.save(prefix + "paper_synth_mstates.txt", raw_ascii);
    runlength_marginals.save(prefix + "paper_synth_runlength.txt", raw_ascii);
    duration_marginals.save(prefix + "paper_synth_duration.txt", raw_ascii);
    implicit_remaining_marginals.save(prefix + "paper_synth_remaining.txt",
            raw_ascii);
    //implicit_duration_marginals.save(prefix + "paper_synth_impli_duration.txt",
    //        raw_ascii);
    return 0;
}
