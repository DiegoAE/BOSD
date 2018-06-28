#include <armadillo>
#include <HMM.hpp>

using namespace arma;
using namespace std;

namespace hsmm {
    
    HMM::HMM(shared_ptr<AbstractEmission> emission, mat transition, vec pi) :
            HSMM(emission, transition, pi, ones<mat>(
            emission->getNumberStates(), 1), 1) {}

};
