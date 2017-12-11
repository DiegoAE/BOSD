cmake_minimum_required(VERSION 3.5)

project(hsmm_inference)

set(CMAKE_CXX_STANDARD 11)

find_package(Armadillo REQUIRED)

include_directories(src)

set(SOURCE_FILES
    src/ForwardBackward.cpp
    examples/FB_example.cpp
    )

add_executable(FB_example ${SOURCE_FILES})

target_link_libraries(FB_example ${ARMADILLO_LIBRARIES})
