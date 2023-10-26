cmake_minimum_required(VERSION 2.8.3)
project(step1)
add_compile_options(-O2 -march=native)
set(SOURCE_FILE main.cpp)

find_package(OpenCV REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(main
    ${SOURCE_FILE}
)

target_link_libraries(main
    ${OpenCV_LIBRARIES}
)

target_link_libraries(main ${JSONCPP_LIBRARIES})