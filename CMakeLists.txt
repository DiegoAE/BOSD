cmake_minimum_required(VERSION 3.5)

project(hidden_semi_markov_models)

set(CMAKE_CXX_STANDARD 11)

find_package(Armadillo REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}
    src/ForwardBackward.cpp
    src/HSMM.cpp
  )

add_executable(hsmm_example examples/dummy_hsmm_example.cpp)

target_link_libraries(hsmm_example ${ARMADILLO_LIBRARIES} ${PROJECT_NAME})
