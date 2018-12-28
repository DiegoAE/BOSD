cmake_minimum_required(VERSION 3.5)

project(hidden_semi_markov_models)

set(CMAKE_CXX_STANDARD 11)

find_package(Armadillo REQUIRED)
find_package(Boost COMPONENTS program_options unit_test_framework REQUIRED)

include_directories(include)
include_directories(${Boost_INCLUDE_DIR})

add_library(${PROJECT_NAME}
    src/emissions.cpp
    src/ForwardBackward.cpp
    src/HSMM.cpp
    src/HMM.cpp
  )

add_executable(hsmm_dummy_gaussian examples/dummy_hsmm_example.cpp)
add_executable(hmm_dummy_gaussian examples/dummy_hmm_example.cpp)

target_link_libraries(hsmm_dummy_gaussian ${PROJECT_NAME} ${ARMADILLO_LIBRARIES})
target_link_libraries(hmm_dummy_gaussian ${PROJECT_NAME} ${ARMADILLO_LIBRARIES})

find_library(ROB_LIB NAMES robotics)
find_library(MLPACK_LIB NAMES mlpack)

add_subdirectory(tests)
enable_testing()
add_test(NAME labels_test COMMAND labels_test WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME FB_test COMMAND FB_test WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME HSMM_test COMMAND HSMM_test WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR})

if (ROB_LIB)
    find_path(ROB_INCLUDES NAMES robotics.hpp)
    include_directories(${ROB_INCLUDES})
    add_executable(promps_hsmm_example examples/promps_hsmm_example.cpp)
    add_executable(promps_hsmm_ball examples/promps_hsmm_ball.cpp)
    add_executable(promps_hsmm_ball_viterbi examples/promps_hsmm_ball_viterbi.cpp)
    add_executable(promps_hsmm_online examples/promps_hsmm_online.cpp)
    add_executable(promps_hsmm_robot examples/promps_hsmm_robot.cpp)
    add_executable(promps_hmm_example examples/promps_hmm_example.cpp)
    add_executable(promps_hmm_robot examples/promps_hmm_robot.cpp)
    add_executable(promps_hsmm_synth_exp examples/promps_hsmm_synth_exp.cpp)
    add_executable(promps_hsmm_sleep_data examples/promps_hsmm_sleep_data.cpp)
    target_link_libraries(promps_hsmm_example
        ${PROJECT_NAME}
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES})
    target_link_libraries(promps_hsmm_ball
        ${PROJECT_NAME}
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES})
    target_link_libraries(promps_hsmm_ball_viterbi
        ${PROJECT_NAME}
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES})
    target_link_libraries(promps_hsmm_online
        ${PROJECT_NAME}
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES})
    target_link_libraries(promps_hsmm_robot
        ${PROJECT_NAME}
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES})
    target_link_libraries(promps_hmm_example
        ${PROJECT_NAME}
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES})
    target_link_libraries(promps_hmm_robot
        ${PROJECT_NAME}
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES})
    target_link_libraries(promps_hsmm_synth_exp
        ${PROJECT_NAME}
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES})
    add_library(Gaussian_emission_promp
        src/Multivariate_Gaussian_emission.cpp
    )
    target_link_libraries(promps_hsmm_sleep_data
        ${PROJECT_NAME}
        Gaussian_emission_promp
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES})
    add_test(NAME ProMPs_test COMMAND ProMPs_test WORKING_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME OnlineHSMMRunlengthBased_MVN_test
        COMMAND OnlineHSMMRunlengthBased_MVN_test WORKING_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR})
else()
    message(WARNING "librobotics was not found. Therefore ProMPs are not supported.")
endif()

if (MLPACK_LIB)
    find_path(MLPACK_INCLUDE_DIR NAMES mlpack/core.hpp mlpack/prereqs.hpp)
    include_directories(${MLPACK_INCLUDE_DIR})
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(flag "OpenMP::OpenMP_CXX")
    else()
        set(flag "")
    endif()
    add_executable(nn_standard_fit examples/NN_standard_fit.cpp)
    target_link_libraries(nn_standard_fit
        ${flag}
        ${MLPACK_LIB}
        ${ARMADILLO_LIBRARIES}
        ${Boost_LIBRARIES}
    )
    if (ROB_LIB)
        add_library(NN_basis_function
            src/NN_basis_function.cpp
        )
        target_link_libraries(NN_basis_function
            ${flag}
            ${ROB_LIB}
            ${MLPACK_LIB}
            ${ARMADILLO_LIBRARIES}
        )
        add_executable(nn_example examples/NN_example.cpp)
        target_link_libraries(nn_example
            ${flag}
            NN_basis_function
            ${PROJECT_NAME}
            ${ROB_LIB}
            ${MLPACK_LIB}
            ${ARMADILLO_LIBRARIES}
            ${Boost_LIBRARIES}
        )
        add_executable(nn_promps_model_from_labels
            examples/NN_promps_hsmm_output_model_from_labels.cpp
        )
        target_link_libraries(nn_promps_model_from_labels
            ${flag}
            NN_basis_function
            ${PROJECT_NAME}
            ${ROB_LIB}
            ${MLPACK_LIB}
            ${ARMADILLO_LIBRARIES}
            ${Boost_LIBRARIES}
        )
    endif()
else()
    message(WARNING "mlpack was not found.")
endif()
