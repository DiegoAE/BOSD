#ifndef NN_BASIS_FUNCTION_H
#define NN_BASIS_FUNCTION_H

#include <armadillo>
#include <json.hpp>
#include <robotics/basis_functions.hpp>


class ScalarNNBasis : public robotics::ScalarBasisFun {
    public:
        ScalarNNBasis(int number_hidden_units) : number_hidden_units_(
                number_hidden_units) {}

        arma::vec eval(double t) const {
            // TODO
            return 0;
        }

        arma::vec deriv(double time, unsigned int order) const {
            // TODO
            return 0;
        }

        unsigned int dim() const {
            return number_hidden_units_;
        }

        nlohmann::json to_stream() const {
            // TODO.
            nlohmann::json ret;
            return ret;
        }

        ~ScalarNNBasis() = default;

    private:
        int number_hidden_units_;
};

#endif
