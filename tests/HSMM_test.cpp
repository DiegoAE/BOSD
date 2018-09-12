#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE HSMM
#include <boost/test/unit_test.hpp>
#include <HSMM.hpp>
#include <memory>

#define EPSILON 1e-6

using namespace arma;
using namespace hsmm;
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

// Emission parameters.
vec means = {0, 5, 10, 15};
vec std_devs =  {0.5, 1.0, 0.1, 2.0};

BOOST_AUTO_TEST_CASE( OnlineHSMM_test ) {
    shared_ptr<AbstractEmissionOnlineSetting> ptr_emission(new DummyGaussianEmission(
            means, std_devs));
    OnlineHSMM dhsmm(ptr_emission, transition, pi, duration, min_duration);
    ivec hiddenStates, hiddenDurations;
    int nSampledSegments = 10;
    field<mat> samples = dhsmm.sampleSegments(nSampledSegments, hiddenStates,
            hiddenDurations);
    int nobs = samples.n_elem;
    for(int i = 0; i < nobs; i++)
        dhsmm.addNewObservation(samples(i));
    for(int i = 0; i < 10; i++) {
        mat next = dhsmm.sampleNextObservation();
        dhsmm.addNewObservation(next);
    }
    cout << "Hidden states " << endl << hiddenStates << endl;
    cout << "Hidden duration " << endl << hiddenDurations << endl;
}

