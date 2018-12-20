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
    int nsamples = 20;
    field<mat> next = dhsmm.sampleNextObservations(nsamples);
    assert(next.n_elem == nsamples);
    cout << "Hidden states " << endl << hiddenStates << endl;
    cout << "Hidden duration " << endl << hiddenDurations << endl;
}

BOOST_AUTO_TEST_CASE( fitting_from_fully_labeled_sequences ) {
    shared_ptr<AbstractEmissionOnlineSetting> ptr_emission(new DummyGaussianEmission(
            means, std_devs));
    OnlineHSMM dhsmm(ptr_emission, transition, pi, duration, min_duration);
    ivec hiddenStates, hiddenDurations;
    int nSampledSegments = 10000;
    field<mat> samples = dhsmm.sampleSegments(nSampledSegments, hiddenStates,
            hiddenDurations);
    vector<int> sequence_states;
    for(int i = 0; i < hiddenStates.n_elem; i++)
        for(int j = 0; j < hiddenDurations(i); j++)
            sequence_states.push_back(hiddenStates(i));\
    ivec hs_seq = conv_to<ivec>::from(sequence_states);
    field<ivec> seqs = {hs_seq};
    dhsmm.setPiFromLabels(seqs);
    dhsmm.setTransitionFromLabels(seqs);
    dhsmm.setDurationFromLabels(seqs);

    // TODO: compare with the actual parameters.
    cout << "Pi:" << endl << dhsmm.pi_ << endl;
    cout << "Transition:" << endl << dhsmm.transition_ << endl;
    cout << "Duration:" << endl << dhsmm.duration_ << endl;
}
