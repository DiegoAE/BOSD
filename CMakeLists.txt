cmake_minimum_required(VERSION 3.5)

project(hidden_semi_markov_models)

set(CMAKE_CXX_STANDARD 11)

find_package(Armadillo REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}
    src/ForwardBackward.cpp
    src/HSMM.cpp
  )

add_executable(hsmm_dummy_gaussian examples/dummy_hsmm_example.cpp)

target_link_libraries(hsmm_dummy_gaussian ${PROJECT_NAME} ${ARMADILLO_LIBRARIES})

find_library(ROB_LIB NAMES robotics)
if (ROB_LIB)
    find_path(ROB_INCLUDES NAMES robotics.hpp)
    include_directories(${ROB_INCLUDES})
    add_executable(promps_hsmm_example examples/promps_hsmm_example.cpp)
    add_executable(promps_hsmm_experiment examples/promps_hsmm_experiment.cpp)
    target_link_libraries(promps_hsmm_example ${PROJECT_NAME} ${ROB_LIB} ${ARMADILLO_LIBRARIES})
    target_link_libraries(promps_hsmm_experiment ${PROJECT_NAME} ${ROB_LIB} ${ARMADILLO_LIBRARIES})
else()
    message(WARNING "librobotics was not found. Therefore ProMPs are not supported.")
endif()
