#include <HMM.hpp>

using namespace arma;
using namespace hsmm;
using namespace std;
using json = nlohmann::json;

int main() {

    // Instantiating the emission process.
    vec means = {0, 5, 10, 15};
    vec std_devs =  {0.5, 1.0, 0.1, 2.0};
    shared_ptr<AbstractEmission> ptr_emission(
            new DummyGaussianEmission(means, std_devs));

    int nstates = means.n_elem;
    vec pi(nstates, fill::eye);
    mat transition = {{0.0, 0.1, 0.4, 0.5},
                      {0.3, 0.0, 0.6, 0.1},
                      {0.2, 0.2, 0.0, 0.6},
                      {0.4, 0.4, 0.2, 0.0}};

    HMM hmm(ptr_emission, transition, pi);

    ivec hs, hd;
    mat obs = hmm.sampleSegments(1500, hs, hd);
    hmm.fit(obs, 10, 1e-5);
    cout << hmm.transition_ << endl;
    cout << hmm.pi_ << endl;
    return 0;
}
