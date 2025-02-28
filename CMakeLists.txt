cmake_minimum_required(VERSION 3.10)
project(ExploringSpaceFillingCurves CXX)

find_package(hdf5 REQUIRED COMPONENTS CXX C)
find_package(HighFive REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${HDF5_INCLUDE_DIRS})


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native -O3 -g3")

add_executable(GSSPC src/main.cpp)


add_executable(GSbase src/base.cpp)
target_link_libraries(GSbase PRIVATE HighFive ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries(GSSPC PRIVATE HighFive ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX)


