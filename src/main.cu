#include "main.h"
#include "mainV3.h"

int main(int argc, char *argv[]) {
    skewAlgorithm=0;
    configureV3(argc, argv,"input/4070000_0310000_010.bil"); // Read input data (model, image...) and parameters
    setResources(dimx,dimy,runMode); // Create cpu and gpu interfaces, set nCPUs, nGPUs
    allocateV3(skewAlgorithm);            // Shared memory allocation and initialization

    std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();
    executeV3();
    std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();
    showResults(skewAlgorithm);

    double t = (double)(t2 - t1).count() / 1000000000;printf("Tiempo: %f\n",t);
    deallocateV3(skewAlgorithm);          // Free memory and interfaces
    return 0;
}
