CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

SET(NATIVE_MODNAME "test_carma_native")
FIND_PACKAGE(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# ##############################################################################
#                                   C++ TESTS                                  #
# ##############################################################################

IF (NOT DEFINED BUILD_NATIVE_TESTS OR BUILD_NATIVE_TESTS)
    ADD_EXECUTABLE(${NATIVE_MODNAME}
        src/test_main.cpp
        src/test_native_arr_to_mat.cpp
        src/test_native_mat_to_arr.cpp
        src/test_native_arraystore.cpp
        src/test_native_roundtrip.cpp
    )

    TARGET_LINK_LIBRARIES(
        ${NATIVE_MODNAME} PUBLIC
        carma
        Catch2::Catch2
        pybind11::embed
        Python3::Python
    )

    TARGET_INCLUDE_DIRECTORIES(${NATIVE_MODNAME}
        PUBLIC
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/python/include>
    )
    INCLUDE(${PROJECT_SOURCE_DIR}/extern/Catch2/contrib/Catch.cmake)
    catch_discover_tests(${NATIVE_MODNAME})

    INSTALL(TARGETS ${NAIVE_MODNAME} RUNTIME DESTINATION tests)
ENDIF ()

# ##############################################################################
#                                 PYTHON TESTS                                 #
# ##############################################################################
IF (NOT DEFINED BUILD_PYTHON_TESTS OR BUILD_PYTHON_TESTS)
    SET(MODNAME "test_carma")

    # ******************************************************************************
    #                                  EXECUTABLE                                  *
    # ******************************************************************************
    pybind11_add_module(${MODNAME}
        MODULE
            src/bindings.cpp
            src/test_nparray.cpp
            src/test_roundtrip.cpp
            src/test_arr_to_mat.cpp
            src/test_mat_to_arr.cpp
            src/test_arraystore.cpp
            src/test_type_caster.cpp
    )
    TARGET_LINK_LIBRARIES(${MODNAME} PUBLIC carma)
    TARGET_INCLUDE_DIRECTORIES(${MODNAME}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tests/src>
    )
    # ******************************************************************************
    #                                   INSTALL                                    *
    # ******************************************************************************
    INSTALL(TARGETS ${MODNAME} DESTINATION tests)
    FILE(GLOB PY_TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.py")
    INSTALL(FILES ${PY_TEST_FILES} DESTINATION tests)

    # ******************************************************************************
    #                                    TESTS                                     *
    # ******************************************************************************
    IF (VALGRIND_TEST_WRAPPER)
        IF (WIN32)
            MESSAGE(FATAL_ERROR "VALGRIND_TEST_WRAPPER is not supported in WIN32 environment")
        ENDIF ()
        FIND_PROGRAM(VALGRIND_EXECUTABLE valgrind)
        IF (NOT VALGRIND_EXECUTABLE)
            MESSAGE(FATAL_ERROR "VALGRIND_TEST_WRAPPER is requested but valgrind executable cannot be found")
        ENDIF ()
        MESSAGE(STATUS "Valgrind test mode enabled")
        MESSAGE(STATUS "Valgrind executable: ${VALGRIND_EXECUTABLE}")
    
        SET(VALGRIND_LOG_FILE /tmp/valgrind-output)
        ADD_TEST(NAME pytest
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMAND ${VALGRIND_EXECUTABLE} --show-leak-kinds=definite --log-file=${VALGRIND_LOG_FILE} ${Python3_EXECUTABLE} -m pytest -vv --valgrind --valgrind-log=${VALGRIND_LOG_FILE}
        )
        SET_PROPERTY(TEST pytest APPEND PROPERTY ENVIRONMENT "PYTHONMALLOC=malloc")

    ELSE()
        ADD_TEST(NAME pytest
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                 COMMAND ${Python3_EXECUTABLE} -m pytest -vv -s
        )
    ENDIF()

    IF (WIN32)
        SET_PROPERTY(TEST pytest
                APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
    ELSE()
        SET_PROPERTY(TEST pytest
                APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
    ENDIF()
ENDIF ()
