#include <armadillo>
#include <HSMM.hpp>
#include <iostream>
#include <memory>
#include <robotics.hpp>

using namespace arma;
using namespace hsmm;
using namespace robotics;
using namespace std;

int main() {
    // Creating a basic ProMP with the desired parameters
    vec mu_w {0, 0, 0, 0};
    mat Sigma_w = 100*eye<mat>(4,4);
    mat Sigma_y = 0.0001*eye<mat>(2,2);
    ProMP promp(mu_w, Sigma_w, Sigma_y);
    cout << "All Good." << endl;
    return 0;
}