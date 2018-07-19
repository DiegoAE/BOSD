#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ProMPs
#include <boost/test/unit_test.hpp>
#include <ProMPs_emission.hpp>

using namespace arma;
using namespace hsmm;
using namespace std;

BOOST_AUTO_TEST_CASE( ProMPs ) {
    int n_basis_functions = 4;
    int njoints = 1;
    int nstates = 5;

    // Setting a third order polynomial basis function for the ProMP
    int polynomial_order = n_basis_functions - 1;
    shared_ptr<ScalarBasisFun> kernel{ new ScalarPolyBasis(polynomial_order)};

    // Instantiating as many ProMPs as hidden states.
    vector<FullProMP> promps;
    for(int i = 0; i < nstates; i++) {
        vec mu_w(n_basis_functions * njoints);
        mu_w.fill(i * 10);
        mat Sigma_w = (i + 1) * eye<mat>(n_basis_functions * njoints,
                    n_basis_functions * njoints);
        mat Sigma_y = 0.0001*eye<mat>(njoints, njoints);
        ProMP promp(mu_w, Sigma_w, Sigma_y);
        FullProMP poly(kernel, promp, njoints);
        promps.push_back(poly);
    }
}
