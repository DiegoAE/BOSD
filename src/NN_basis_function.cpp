#include <NN_basis_function.hpp>

using namespace arma;
using namespace mlpack::ann;
using namespace std;

namespace hsmm {
    
    ScalarNNBasis::ScalarNNBasis(int number_hidden_units) :
            number_hidden_units_(number_hidden_units) {
        neural_net_.Add<Linear<> >(1, number_hidden_units_);
        neural_net_.Add<SigmoidLayer<> >();
        neural_net_.Add<Linear<> >(10, number_hidden_units_);
    }

    vec ScalarNNBasis::eval(double t) const {
        vec ret;
        return ret;
    }

    vec ScalarNNBasis::deriv(double time, unsigned int order) const {
        vec ret;
        return ret;
    }

    unsigned int ScalarNNBasis::dim() const {
        return number_hidden_units_;
    }

    nlohmann::json ScalarNNBasis::to_stream() const {
        nlohmann::json ret;
        return ret;
    }

};
