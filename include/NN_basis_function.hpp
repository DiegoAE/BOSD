#ifndef NN_BASIS_FUNCTION_H
#define NN_BASIS_FUNCTION_H

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <armadillo>
#include <json.hpp>
#include <robotics/basis_functions.hpp>

namespace hsmm {

    typedef mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>,
            mlpack::ann::RandomInitialization> NNmodel;

    class ScalarNNBasis : public robotics::ScalarBasisFun {
        public:
            ScalarNNBasis(int number_hidden_units, int njoints);

            NNmodel& getNeuralNet() const;

            void setNeuralNet(NNmodel &neural_net);

            arma::vec eval(double t) const;

            // TODO.
            arma::vec deriv(double time, unsigned int order) const;

            unsigned int dim() const;

            nlohmann::json to_stream() const;

            ~ScalarNNBasis() = default;

        private:
            int number_hidden_units_;

            // TODO. Get this number from the neural network itself.
            int number_layers_ = 3;
            mutable NNmodel neural_net_;
    };

};

#endif
