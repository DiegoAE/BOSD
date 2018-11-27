#include <NN_basis_function.hpp>

using namespace arma;
using namespace mlpack::ann;
using namespace std;

namespace hsmm {
    
    ScalarNNBasis::ScalarNNBasis(ivec hidden_units_per_layer, int njoints) :
            hidden_units_per_layer_(hidden_units_per_layer),
            neural_net_outputs_(njoints) {
        assert(hidden_units_per_layer_.n_elem > 0);
        neural_net_.Add<Linear<> >(1, hidden_units_per_layer_(0));
        neural_net_.Add<SigmoidLayer<> >();
        for(int i = 1; i < hidden_units_per_layer_.n_elem; i++) {
            int last_number_units = hidden_units_per_layer_(i - 1);
            int current_number_units = hidden_units_per_layer_(i);
            neural_net_.Add<Linear<> >(last_number_units, current_number_units);
            neural_net_.Add<SigmoidLayer<> >();
        }

        // Connecting the last hidden layer with the output layer.
        output_layer_ = new Linear<>(dim(), neural_net_outputs_);
        neural_net_.Add(output_layer_);
    }

    NNmodel& ScalarNNBasis::getNeuralNet() const {
        return neural_net_;
    }

    int ScalarNNBasis::getNumberLayers() const {
        return hidden_units_per_layer_.n_elem + 2;
    }

    pair<mat, vec> ScalarNNBasis::getOutputLayerParams() const {
        mat params = output_layer_->Parameters();
        mat last_layer_weigths = mat(params.memptr(), neural_net_outputs_,
                dim());
        vec last_layer_bias = mat(params.memptr() + last_layer_weigths.n_elem,
                neural_net_outputs_, 1);
        return make_pair(last_layer_weigths, last_layer_bias);
    }

    void ScalarNNBasis::setNeuralNet(NNmodel &neural_net) {
        neural_net_ = neural_net;
    }

    vec ScalarNNBasis::eval(double t) const {
        mat input = {t};
        mat output;

        // Eval up to the second-to-last layer.
        neural_net_.Forward(input, output, 0, getNumberLayers() - 2);
        return output;
    }

    vec ScalarNNBasis::deriv(double time, unsigned int order) const {
        vec ret;
        throw std::logic_error("Not implemented yet.");
        return ret;
    }

    unsigned int ScalarNNBasis::dim() const {
        return hidden_units_per_layer_(hidden_units_per_layer_.n_elem - 1);
    }

    nlohmann::json ScalarNNBasis::to_stream() const {
        nlohmann::json ret;
        throw std::logic_error("Not implemented yet.");
        return ret;
    }

};
