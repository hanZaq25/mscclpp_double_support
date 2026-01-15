#include <mpi.h>
#include <cuda_runtime.h>
#include <mscclpp/nccl.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(cmd) do {                                \
  cudaError_t e = cmd;                                      \
  if (e != cudaSuccess) {                                   \
    std::cerr << "CUDA error: " << cudaGetErrorString(e)   \
              << " at " << __FILE__ << ":" << __LINE__      \
              << std::endl;                                 \
    exit(1);                                                \
  }                                                         \
} while(0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(rank));
    
    std::cout << "Rank " << rank << ": Initializing NCCL..." << std::endl;
    
    // Initialize NCCL
    ncclComm_t comm;
    ncclUniqueId id;
    if (rank == 0) {
        ncclResult_t res = ncclGetUniqueId(&id);
        std::cout << "Rank 0: ncclGetUniqueId returned " << res << std::endl;
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    ncclResult_t res = ncclCommInitRank(&comm, size, id, rank);
    std::cout << "Rank " << rank << ": ncclCommInitRank returned " << res 
              << " (" << ncclGetErrorString(res) << ")" << std::endl;
    
    if (res != ncclSuccess) {
        std::cerr << "Rank " << rank << ": Failed to initialize NCCL" << std::endl;
        return 1;
    }
    
    // Allocate data
    const int N = 1024;
    double *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(double)));
    
    // Initialize input
    std::vector<double> h_input(N, rank + 1.0);
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Verify input was copied correctly
    std::vector<double> h_verify(N);
    CUDA_CHECK(cudaMemcpy(h_verify.data(), d_input, N * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << "Rank " << rank << ": Input verification: " << h_verify[0] 
              << " (expected " << (rank + 1.0) << ")" << std::endl;
    
    // AllReduce with double
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    std::cout << "Rank " << rank << ": Calling ncclAllReduce with ncclFloat64 (value=" 
              << ncclFloat64 << ")..." << std::endl;
    
    ncclResult_t result = ncclAllReduce(
        d_input, d_output, N, 
        ncclFloat64,  // Double precision
        ncclSum, 
        comm, stream
    );
    
    std::cout << "Rank " << rank << ": ncclAllReduce returned " << result 
              << " (" << ncclGetErrorString(result) << ")" << std::endl;
    
    if (result != ncclSuccess) {
        std::cerr << "Rank " << rank << ": ncclAllReduce failed!" << std::endl;
        return 1;
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Verify result
    std::vector<double> h_output(N);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Show first 10 values
    std::cout << "Rank " << rank << ": First 10 output values: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Expected: sum of (1 + 2 + ... + size) = size*(size+1)/2
    double expected = size * (size + 1) / 2.0;
    bool correct = (std::abs(h_output[0] - expected) < 1e-9);
    
    std::cout << "Rank " << rank << ": Result = " << h_output[0] 
              << ", Expected = " << expected 
              << (correct ? " ✓ PASS" : " ✗ FAIL") << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    MPI_Finalize();
    
    return correct ? 0 : 1;
}