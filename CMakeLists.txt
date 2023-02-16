cmake_minimum_required(VERSION 3.16)
project(skewEngine CUDA CXX)

message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(OMPFLAG "-fopenmp")
set(MYCXXFLAGS "${OMPFLAG} -DGENERIC")
set(CUDA_VERBOSE_BUILD ON)


if (${HOST} STREQUAL "cactus")
    message("CMAKE para Cactus")
    set(OMPFLAG "/openmp")
    set(MYCXXFLAGS "${OMPFLAG} -DCACTUS")
    set(MYCUDAFLAGS "${OMPFLAG},-DCACTUS")
    set(CMAKE_CUDA_ARCHITECTURES "86")
    set(CMAKE_CUDA_COMPILER  "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\bin\\nvcc.exe")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "D:\\onedrive\\proyectos\\skewEngine")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "D:\\onedrive\\proyectos\\skewEngine")
endif()

if (${HOST} STREQUAL "aloe")
    message("CMAKE para Aloe")
    set(OMPFLAG "-fopenmp")
    set(MYCXXFLAGS "-DALOE -fpermissive")
    set(MYCUDAFLAGS "${OMPFLAG},-DALOE")
    set(CMAKE_CUDA_ARCHITECTURES "30")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
    set(Boost_INCLUDE_DIR "/usr/local/include/")
    set(Boost_LIBRARY_DIR "/usr/local/lib/")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "/users/felipe")
endif()

set(ENV{CUDACXX}  "${CMAKE_CUDA_COMPILER}")



set (HEADERS
        main.h
        GpuSolver.cuh
        defaults.h
        Constants.h
        Global.h
        Lodepng.h
        color.h
        mainPath.h
        drawing.h
        auxf.h
        auxf.cuh
        skewEngine.cuh
        )

set (SOURCES
        src/main.cu
        src/GpuSolver.cu
        src/Global.cpp
        src/third-party/Lodepng.cpp
        src/color.cpp
        src/drawing.cpp
        src/auxf.cpp
        src/auxf.cu
        src/skewEngine.cu
#        ${HEADERS}
        )

find_package(OpenMP)
include_directories(SYSTEM ./src)
add_executable(skewEngine ${SOURCES})
target_include_directories(skewEngine
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/third-party
        )




set(CMAKE_CUDA_FLAGS_DEBUG  "--expt-relaxed-constexpr -g -Xcompiler  ${MYCUDAFLAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE   "--expt-relaxed-constexpr -O5 -Xcompiler ${MYCUDAFLAGS}")

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}  ${MYCXXFLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${MYCXXFLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
