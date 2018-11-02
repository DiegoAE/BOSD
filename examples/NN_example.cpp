#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <NN_emission.hpp>

using namespace mlpack::ann;

int main() {
    std::vector<FFN<> > ffns;
    FFN<> model;
    model.Add<Linear<> >(10, 8);
    model.Add<SigmoidLayer<> >();
    model.Add<Linear<> >(8, 3);
    model.Add<LogSoftMax<> >();
    ffns.push_back(model);
    hsmm::NNEmission e(ffns, 1);
    std::cout << "OK" << std::endl;
    return 0;
}

