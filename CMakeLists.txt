cmake_minimum_required(VERSION 3.16)

message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
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
    set(CMAKE_CUDA_COMPILER  "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin\\nvcc.exe")
    #set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "D:\\")
    #set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "D:\\onedrive\\proyectos\\skewEngine")
endif()

if (${HOST} STREQUAL "aloe")
    message("CMAKE para Aloe")
    set(OMPFLAG "-fopenmp")
    set(MYCXXFLAGS "-DALOE -fpermissive")
    set(MYCUDAFLAGS "${OMPFLAG},-DALOE -fpermissive")
    set(CMAKE_CUDA_ARCHITECTURES "30")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
    set(Boost_INCLUDE_DIR "/usr/local/include/")
    set(Boost_LIBRARY_DIR "/usr/local/lib/")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "/users/felipe")
endif()

if (${HOST} STREQUAL "picasso")
    message("CMAKE para Picasso")
    set(CMAKE_CUDA_STANDARD 14)
    set(OMPFLAG "-fopenmp")
    set(MYCXXFLAGS "${OMPFLAG} -D PICASSO")
    set(CMAKE_CUDA_ARCHITECTURES "80")
    set(CMAKE_CUDA_COMPILER "/mnt/home/soft/pgi/programas/linux86-64/nvhpc/Linux_x86_64/20.9/cuda/11.0/bin/nvcc")
    set(ENV{CUDACXX}  "${CMAKE_CUDA_COMPILER}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "/mnt/home/users/ac/felipe")
endif()



set(ENV{CUDACXX}  "${CMAKE_CUDA_COMPILER}")

project(skewEngine CUDA CXX)


set (HEADERS
        src/main.h
        src/kernelRADN.h
        src/kernelBLUR.h
        src/kernelSDEM.h
        src/kernelUNIT.h
        src/defaults.h
        src/global.h
        src/color.h
        src/helper.h
        src/skewEngine.cuh
        )


set (SOURCES
        src/main.cu
        src/global.cpp
        src/third-party/Lodepng.cpp
        src/color.cpp
        ${HEADERS}
        )

find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)

message("OpenCL: ${OpenCL_INCLUDE_DIRS} ${OpenCL_LIBRARY}")


#include_directories(SYSTEM ./src)
add_executable(skewEngine ${SOURCES})
add_library(sk_engine SHARED ${SOURCES})
target_include_directories(skewEngine
        INTERFACE
        third-party
        )

# set_source_files_properties(src/main.cu PROPERTIES OBJECT_DEPENDS src/kernelRADN.h)

if (NOT ${HOST} STREQUAL "cactus")
    target_link_libraries(skewEngine stdc++fs cufft)
    target_link_libraries(sk_engine stdc++fs cufft)
else()
    link_directories(${OpenCL_LIBRARY})
    include_directories(${OpenCL_INCLUDE_DIRS})
    target_link_libraries(skewEngine cufft)
endif()

target_link_libraries(skewEngine OpenCL::OpenCL)

set(CMAKE_CUDA_FLAGS_DEBUG  "--expt-relaxed-constexpr -g -Xcompiler  ${MYCUDAFLAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE   "--expt-relaxed-constexpr -O5 -Xcompiler ${MYCUDAFLAGS}")

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}  ${MYCXXFLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${MYCXXFLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

