cmake_minimum_required(VERSION 3.16)
project(cjong4_move_evaluator VERSION 0.1.0 LANGUAGES C)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(CJ4ME_BUILD_TESTS "Build cj4me tests" OFF)
option(CJ4ME_BUILD_TOOLS "Build cj4me command line tools" OFF)
set(CJONG4_SOURCE_DIR "" CACHE PATH "Path to a cjong4 source tree")

if(NOT TARGET cjong4::cj4)
    if(CJONG4_SOURCE_DIR)
        set(BUILD_TESTS OFF)
        add_subdirectory("${CJONG4_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/cjong4")
    else()
        find_package(cjong4 3.2.0 CONFIG REQUIRED)
    endif()
endif()

add_library(cj4me STATIC
    src/dataset.c
    src/evaluator.c
    src/feature.c
    src/inference_f32.c
    src/inference_i8.c
    src/model_io.c
    src/rng.c
    src/selfplay.c
)
add_library(cjong4_move_evaluator::cj4me ALIAS cj4me)

target_include_directories(cj4me
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(cj4me PUBLIC cjong4::cj4)

if(MSVC)
    target_compile_options(cj4me PRIVATE /W4)
else()
    target_compile_options(cj4me PRIVATE -Wall -Wextra -Wpedantic)
endif()

function(cj4me_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

if(CJ4ME_BUILD_TOOLS)
    add_executable(cj4me_generate tools/cj4me_generate.c)
    target_link_libraries(cj4me_generate PRIVATE cj4me)
    target_include_directories(cj4me_generate PRIVATE src)
    cj4me_enable_warnings(cj4me_generate)

    add_executable(cj4me_evaluate tools/cj4me_evaluate.c)
    target_link_libraries(cj4me_evaluate PRIVATE cj4me)
    cj4me_enable_warnings(cj4me_evaluate)
endif()

if(CJ4ME_BUILD_TESTS)
    enable_testing()
    add_executable(cj4me_tests
        tests/test_main.c
        tests/test_dataset.c
        tests/test_evaluator.c
        tests/test_feature.c
        tests/test_model.c
        tests/test_rng.c
    )
    target_link_libraries(cj4me_tests PRIVATE cj4me)
    cj4me_enable_warnings(cj4me_tests)
    if(MSVC)
        target_compile_options(cj4me_tests PRIVATE /UNDEBUG)
    else()
        target_compile_options(cj4me_tests PRIVATE -UNDEBUG)
    endif()
    add_test(NAME cj4me_tests COMMAND cj4me_tests)
endif()

install(TARGETS cj4me
    EXPORT cjong4MoveEvaluatorTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(CJ4ME_CMAKE_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/cjong4_move_evaluator")
install(EXPORT cjong4MoveEvaluatorTargets
    FILE cjong4MoveEvaluatorTargets.cmake
    NAMESPACE cjong4_move_evaluator::
    DESTINATION ${CJ4ME_CMAKE_INSTALL_DIR}
)

configure_package_config_file(
    cmake/cjong4_move_evaluatorConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/cjong4_move_evaluatorConfig.cmake"
    INSTALL_DESTINATION ${CJ4ME_CMAKE_INSTALL_DIR}
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cjong4_move_evaluatorConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cjong4_move_evaluatorConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cjong4_move_evaluatorConfigVersion.cmake"
    DESTINATION ${CJ4ME_CMAKE_INSTALL_DIR}
)
