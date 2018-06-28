#ifndef HMM_H
#define HMM_H

#include <armadillo>
#include <HSMM.hpp>

namespace hsmm {

    class HMM : public HSMM {
        public:
            HMM(std::shared_ptr<AbstractEmission> emission,
                    arma::mat transition, arma::vec pi);
    };

};

#endif
