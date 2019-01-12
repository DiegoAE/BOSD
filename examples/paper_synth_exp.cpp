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
#define NDUR 10
#define MINDUR 30

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

class ToyEmission : public AbstractEmissionOnlineSetting {
    public:
        ToyEmission(vector<function<vec(double)>> states, int ndim) : states_(
                states), AbstractEmissionOnlineSetting(states.size(), ndim) {
            output_gaussian_stddev_ = ones<mat>(getNumberStates(),
                    getDimension());
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
                vec diff = obs(i) - states_.at(state)(t(i));
                for(int j = 0; j < diff.n_elem; j++)
                    ret += gaussianlogpdf_(diff(j), 0,
                            output_gaussian_stddev_(state, j));
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

vec simple(double t) {
    assert(t <= 1.0 && t >= 0); 
    return ones<vec>(NDIM) * t;
}

vec simple2(double t) {
    assert(t <= 1.0 && t >= 0); 
    return ones<vec>(NDIM) * t * 2;
}

int main(int arc, char* argv[]) {
    function<vec(double)> f1 = simple;
    function<vec(double)> f2 = simple2;
    vector<function<vec(double)>> states = {simple, simple2};
    shared_ptr<ToyEmission> ptr_emission(new ToyEmission(states, NDIM));

    vec pi(states.size());
    mat transition(states.size(), states.size());
    mat duration(states.size(), NDUR);
    pi.fill(1.0 / states.size());
    transition.fill(1.0 / states.size());
    duration.fill(1.0 / NDUR);
    OnlineHSMM online_toy_model(static_pointer_cast<
            AbstractEmissionOnlineSetting>(ptr_emission), transition, pi,
            duration, MINDUR);
    int nsegments = 10;
    ivec hs, hd;
    field<mat> toy_seq = online_toy_model.sampleSegments(nsegments, hs, hd);
    imat vit_mat = join_horiz(hs, hd);
    mat output = fieldToMat(NDIM, toy_seq);
    output.save("paper_synth_exp.txt", raw_ascii);
    return 0;
}
