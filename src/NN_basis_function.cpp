#include <NN_basis_function.hpp>

using namespace arma;
using namespace mlpack::ann;
using namespace std;

namespace hsmm {
    
    ScalarNNBasis::ScalarNNBasis(int number_hidden_units, int njoints) :
            number_hidden_units_(number_hidden_units) {
        neural_net_.Add<Linear<> >(1, number_hidden_units_);
        neural_net_.Add<SigmoidLayer<> >();
        neural_net_.Add<Linear<> >(number_hidden_units_, njoints);
    }

    NNmodel& ScalarNNBasis::getNeuralNet() const {
        return neural_net_;
    }

    void ScalarNNBasis::setNeuralNet(NNmodel &neural_net) {
        neural_net_ = neural_net;
    }

    // TODO. Eval up to the second-to-last layer.
    vec ScalarNNBasis::eval(double t) const {
        mat input = {t};
        mat output;
        neural_net_.Predict(input, output);
        return output;
    }

    vec ScalarNNBasis::deriv(double time, unsigned int order) const {
        vec ret;
        throw std::logic_error("Not implemented yet.");
        return ret;
    }

    unsigned int ScalarNNBasis::dim() const {
        return number_hidden_units_;
    }

    nlohmann::json ScalarNNBasis::to_stream() const {
        nlohmann::json ret;
        throw std::logic_error("Not implemented yet.");
        return ret;
    }

};
