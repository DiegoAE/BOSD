### Online Hidden-semi Markov Models for Bayesian Online Detection and Prediction of Change Points

Implementation of Online HSMMs which eases the implementation of custom emission processes. In particular, emission processes which jointly depend on the hidden state and duration are supported.

There is also a python script for plotting the inferences drawn from the model.

#### Requirements

The framework requires the following ubuntu packages:

```
sudo apt-get install libboost-all-dev nlohmann-json-dev libarmadillo-dev cmake
```

If the ProMPs functionality is also wanted then you'll need to install TODO.

#### Building

Follow the standard procedure for building cmake projects. Once you are in repository main directory type:

```
mkdir build
cd build
cmake ..
make
```
The latter will only build executables for which the dependencies are fulfilled. Make sure your mlpack and ProMPs installations are discoverable by cmake if you want to build their respective examples.

#### Examples

Examples of HSMMs with different emission processes can be found in the examples folder distributed with the code.

#### References

Please cite the paper *Bayesian Online Detection and Prediction of Change Points* (https://arxiv.org/abs/1902.04524) if you find this work useful.
