
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
add_executable(labels_test
  labels_test.cpp
  )

target_link_libraries(labels_test
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${PROJECT_NAME}
  ${ARMADILLO_LIBRARIES}
  )
