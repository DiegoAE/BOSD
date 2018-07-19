
add_executable(labels_test
    labels_test.cpp
  )

add_executable(FB_test
    FB_test.cpp
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

if (ROB_LIB)
    add_executable(ProMPs_test ProMPs_test.cpp )
    target_link_libraries(ProMPs_test
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        ${PROJECT_NAME}
        ${ROB_LIB}
        ${ARMADILLO_LIBRARIES}
        )
endif()
