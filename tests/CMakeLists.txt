
add_executable(labels_test
    labels_test.cpp
  )

add_executable(FB_test
    FB_test.cpp
  )

add_executable(HSMM_test
    HSMM_test.cpp
  )

target_link_libraries(labels_test
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${PROJECT_NAME}
  ${ARMADILLO_LIBRARIES}
  )

target_link_libraries(FB_test
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${PROJECT_NAME}
  ${ARMADILLO_LIBRARIES}
  )

target_link_libraries(HSMM_test
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${PROJECT_NAME}
  ${ARMADILLO_LIBRARIES}
  )

if (ROB_LIB)
    find_path(ROB_INCLUDES NAMES robotics.hpp)
    include_directories(${ROB_INCLUDES})
    add_executable(ProMPs_test ProMPs_test.cpp )
    target_link_libraries(ProMPs_test
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        ${PROJECT_NAME}
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES}
        )
    add_executable(OnlineHSMMRunlengthBased_MVN_test
        OnlineHSMMRunlengthBased_MVN_test.cpp
      )
    target_link_libraries(OnlineHSMMRunlengthBased_MVN_test
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        ${PROJECT_NAME}
        Gaussian_emission_promp
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES}
        )
endif()
