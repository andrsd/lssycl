cmake_minimum_required(VERSION 3.16)

project(lssycl
    VERSION 0.0.1
    LANGUAGES CXX
)

if (APPLE)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

find_package(fmt 8.0 REQUIRED)
find_package(AdaptiveCpp REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
add_sycl_to_target(TARGET ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/contrib
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        fmt::fmt
)
