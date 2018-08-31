#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ProMPs
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <ProMPs_emission.hpp>

#define EPSILON 1e-6

using namespace arma;
using namespace hsmm;
using namespace std;


ProMPsEmission getExampleProMP() {
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

    ProMPsEmission emission(promps);
    return emission;
}

BOOST_AUTO_TEST_CASE( ProMPs ) {

    // Creating the ProMP emission.
    ProMPsEmission emission = getExampleProMP();

    for(int i = 0; i < emission.getNumberStates(); i++) {
        for(int size = 1; size < 100; size += 10) {
            field<mat> obs = emission.sampleFromState(i, size);
            double loglikelihood = emission.loglikelihood(i, obs);
            double Iloglikelihood = emission.informationFilterLoglikelihood(i,
                obs);
            BOOST_CHECK(fabs(loglikelihood - Iloglikelihood) < EPSILON);
        }
    }

    // Comparing the running time.
    int benchmark_size = 200;
    for(int i = 0; i < emission.getNumberStates(); i++) {
        auto t1 = chrono::high_resolution_clock::now();
        auto sample = emission.sampleFromState(i, benchmark_size);
        double kf_loglikelihood = emission.loglikelihood(i, sample);
        auto t2 = chrono::high_resolution_clock::now();
        double if_loglikelihood = emission.informationFilterLoglikelihood(i,
                sample);
        auto t3 = chrono::high_resolution_clock::now();
        auto elapsed_kf = chrono::duration_cast<chrono::milliseconds>(
                t2 - t1).count();
        auto elapsed_if = chrono::duration_cast<chrono::milliseconds>(
                t3 - t2).count();
        cout << "Elapsed KF: " << elapsed_kf << " Elapsed IF: " <<
                elapsed_if << endl;
        BOOST_CHECK(fabs(kf_loglikelihood - if_loglikelihood)
                < EPSILON);
    }

    // Checking the missing output handling.
    for(int i = 0; i < emission.getNumberStates(); i++) {
        int size = 100;
        field<mat> obs1 = emission.sampleFromState(i, size);
        int missing_from = 50;
        for(int j = missing_from; j < obs1.n_elem; j++)
            obs1(j).reset();
        // Making sure it doesn't fail. TODO: compare with other thing.
        double ll1 = emission.loglikelihood(i, obs1);
    }
}
