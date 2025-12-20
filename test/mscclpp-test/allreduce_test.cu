// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <cstring>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/packet_device.hpp>
#include <vector>

#include "common.hpp"

#define BLOCKS_PER_PEER 1

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::PortChannel> constDevFstRoundChans[16];
__constant__ DeviceHandle<mscclpp::PortChannel> constDevSndRoundChans[16];

__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemInPlaceChans[8];
__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemOutOfPlaceChans[8];
__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemOutOfPlaceGetChans[8];
__device__ uint64_t globalFlag;

// TODO(chhwang): need an interface for this.
static void* inputBuff = nullptr;
static void* resultBuff = nullptr;
static void* scratchBuff = nullptr;
static void* scratchPacketBuff = nullptr;
static void* putPacketBuff = nullptr;
static void* getPacketBuff = nullptr;

struct Chunk {
  size_t offset;
  size_t size;
};

__host__ __device__ Chunk getChunk(size_t dataCount, size_t numChunks, size_t chunkIdx) {
  size_t remainder = dataCount % numChunks;
  size_t smallChunkSize = dataCount / numChunks;
  size_t largeChunkSize = smallChunkSize + 1;
  size_t numRemainedLargeChunks = chunkIdx < remainder ? remainder - chunkIdx : 0;
  size_t offset = (remainder - numRemainedLargeChunks) * largeChunkSize +
                  (chunkIdx > remainder ? chunkIdx - remainder : 0) * smallChunkSize;
  return Chunk{offset, chunkIdx < remainder ? largeChunkSize : smallChunkSize};
}

__forceinline__ __device__ void vectorSum(double* dst, double* src, size_t nElem, int blockId, int nBlocks) {
  // We process 2 doubles per vector to maintain 128-bit width (16 bytes)
  size_t nDouble2 = nElem / 2;
  size_t nLastDoubles = nElem % 2;

  // Cast to CUDA vector types
  double2* dst2 = (double2*)dst;
  double2* src2 = (double2*)src;

  // Main vectorized loop
  for (size_t i = threadIdx.x + blockId * blockDim.x; i < nDouble2; i += blockDim.x * nBlocks) {
    // double2 only has .x and .y components
    dst2[i].x += src2[i].x;
    dst2[i].y += src2[i].y;
  }

  // Handle the tail (if nElem is odd)
  if (nLastDoubles > 0) {
    // Revert to scalar pointer arithmetic
    double* dstLast = dst + nDouble2 * 2;
    double* srcLast = src + nDouble2 * 2;
    
    for (size_t i = threadIdx.x + blockId * blockDim.x; i < nLastDoubles; i += blockDim.x * nBlocks) {
      dstLast[i] += srcLast[i];
    }
  }
}

__forceinline__ __device__ void vectorSum(double* dst, double* src, size_t nElem) {
  vectorSum(dst, src, nElem, blockIdx.x, gridDim.x);
}

__device__ void vectorSumSingleBlock(double* dst, double* src, size_t nElem) {
  for (size_t i = threadIdx.x; i < nElem; i += blockDim.x) {
    dst[i] += src[i];
  }
}

__device__ mscclpp::DeviceSyncer deviceSyncer;
__device__ mscclpp::DeviceSyncer allGatherDeviceSyncer;
__device__ mscclpp::DeviceSyncer reduceScatterDeviceSyncer;
__device__ mscclpp::DeviceSyncer ibDeviceSyncer;

// Reduce-Scatter among the GPUs that belong to the same node using the Ring Algo.
__device__ void localReduceScatter(double* buff, double* scratch, int rank, int nRanksPerNode, int startChunkIndex,
                                   size_t offsetInChunk, size_t chunkSize, size_t nelems) {
  if (nRanksPerNode == 1) {
    return;
  }
  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int startRankInNode = (rank / nRanksPerNode) * nRanksPerNode;
  int rankIndexInNode = rank % nRanksPerNode;

  for (int i = 1; i < nRanksPerNode; ++i) {
    int remoteSendToRank = (rank + i) % nRanksPerNode + startRankInNode;
    int remoteRecvFromRank = (rank + nRanksPerNode - i) % nRanksPerNode + startRankInNode;
    int peerSendId = (remoteSendToRank < rank) ? remoteSendToRank : remoteSendToRank - 1;
    int peerRecvId = (remoteRecvFromRank < rank) ? remoteRecvFromRank : remoteRecvFromRank - 1;

    DeviceHandle<mscclpp::PortChannel>& devFstSendChan = constDevFstRoundChans[peerSendId];
    DeviceHandle<mscclpp::PortChannel>& devFstRecvChan = constDevFstRoundChans[peerRecvId];
    size_t srcOffset =
        (((rankIndexInNode + i) % nRanksPerNode + startChunkIndex) * chunkSize + offsetInChunk) * sizeof(double);
    size_t dstOffset = rank * chunkSize * sizeof(double);

    if (i == 1) {
      if (isComm) {
        devFstSendChan.putWithSignal(dstOffset, srcOffset, nelems * sizeof(double));
      }
    } else {
      int pre = i - 1;
      int preRemoteRecvFromRank = (rank + nRanksPerNode - pre) % nRanksPerNode + startRankInNode;
      int prePeerRecvId = (preRemoteRecvFromRank < rank) ? preRemoteRecvFromRank : preRemoteRecvFromRank - 1;

      // overlap communication and computation
      DeviceHandle<mscclpp::PortChannel>& preDevFstRecvChan = constDevFstRoundChans[prePeerRecvId];
      if (isComm) {
        preDevFstRecvChan.wait();
        devFstSendChan.putWithSignal(dstOffset, srcOffset, nelems * sizeof(double));
      }

      deviceSyncer.sync(gridDim.x);
      size_t offset = ((startChunkIndex + rankIndexInNode) * chunkSize + offsetInChunk) * sizeof(double);
      size_t scratchOffset = preRemoteRecvFromRank * chunkSize * sizeof(double);
      double* dst = (double*)((char*)buff + offset);
      double* src = (double*)((char*)scratch + scratchOffset);
      vectorSum(dst, src, nelems);
    }
    // for last iteration, wait for the last send
    if (i == nRanksPerNode - 1) {
      if (isComm) {
        devFstRecvChan.wait();
      }
      deviceSyncer.sync(gridDim.x);
      size_t offset = ((startChunkIndex + rankIndexInNode) * chunkSize + offsetInChunk) * sizeof(double);
      size_t scratchOffset = remoteRecvFromRank * chunkSize * sizeof(double);
      double* dst = (double*)((char*)buff + offset);
      double* src = (double*)((char*)scratch + scratchOffset);
      vectorSum(dst, src, nelems);
    }
  }
}

// has inter node communication.
__device__ void reduceScatter(double* buff, double* scratch, int rank, int nRanksPerNode, int worldSize,
                              size_t nelems  // must be divisible by 3
) {
  // this reduce-scatter algorithm works as follows:
  // Step 1: each node does a local reduce-scatter on peer node data chunks with 1/pipeline portion of chunk data. For
  // example, 2 nodes and each node has 2 ranks. rank 0 and rank 1 perform reduce-scatter on chunk 2 and chunk 3, with
  // 1/pipeline portion of the data.
  // Step 2: each node does a local reduce-scatter on peers data chunks with (pipeline-1)/pipeline portion of chunk
  // data. Meanwhile, exchange the reduced data of the previous step with its cross-node neighbor (same local rank
  // number on the other node) via IB. Then performs a reduce operation.
  // Step 3:  each node does a local reduce-scatter on local ranks, meanwhile exchange the reduced data of the previous
  // step with its cross-node neighbor (same local rank number on the other node) via IB. Then performs a reduce
  // operation.
  int pipelineSize = 3;
  const size_t chunkSize = nelems / worldSize;
  int peerRank = (rank + nRanksPerNode) % worldSize;
  int peerNodeId = peerRank / nRanksPerNode;
  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int peer = (peerRank < rank) ? peerRank : peerRank - 1;
  DeviceHandle<mscclpp::PortChannel>& portChan = constDevFstRoundChans[peer];
  if (peerNodeId == rank / nRanksPerNode) {
    localReduceScatter(buff, scratch, rank, nRanksPerNode, 0, 0, chunkSize, chunkSize);
    return;
  }

  // step 1: local reduce
  int startChunkIndex = peerNodeId * nRanksPerNode;
  localReduceScatter(buff, scratch, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize / pipelineSize);
  deviceSyncer.sync(gridDim.x);

  // step 2: local reduce and exchange data with neighbor
  if (isComm) {
    size_t offset = (peerRank * chunkSize) * sizeof(double);
    // opposite side
    portChan.putWithSignal(offset, (chunkSize / pipelineSize * sizeof(double)));
  }
  localReduceScatter(buff, scratch, rank, nRanksPerNode, startChunkIndex, chunkSize / pipelineSize, chunkSize,
                     2 * chunkSize / pipelineSize);
  if (isComm) {
    portChan.wait();
  }
  deviceSyncer.sync(gridDim.x);
  // reduce data received from peer to related rank
  size_t offset = rank * chunkSize * sizeof(double);
  double* dst = (double*)((char*)buff + offset);
  double* src = (double*)((char*)scratch + offset);
  vectorSum(dst, src, chunkSize / pipelineSize);
  if (isComm) {
    portChan.flush();
  }
  deviceSyncer.sync(gridDim.x);

  // step 3: local reduce and exchange data with neighbor
  startChunkIndex = (rank / nRanksPerNode) * nRanksPerNode;
  if (isComm) {
    size_t offset = (peerRank * chunkSize + chunkSize / pipelineSize) * sizeof(double);
    portChan.putWithSignal(offset, (pipelineSize - 1) * chunkSize / pipelineSize * sizeof(double));
  }
  localReduceScatter(buff, scratch, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize);
  if (isComm) {
    portChan.wait();
  }
  deviceSyncer.sync(gridDim.x);
  // reduce to related rank
  offset = (rank * chunkSize + chunkSize / pipelineSize) * sizeof(double);
  dst = (double*)((char*)buff + offset);
  src = (double*)((char*)scratch + offset);
  vectorSum(dst, src, 2 * chunkSize / pipelineSize);
  if (isComm) {
    portChan.flush();
  }
}

//intra node + Ring Algo
// Run with a single thread only.
__device__ void localAllGather(int rank, int nRanksPerNode, uint64_t offset, uint64_t size) {
  // this allgather algorithm works as follows:
  // Step 1: GPU rank i sends data to GPU rank (i+1) % nranksPerNode
  // and waits for data from GPU rank (i-1) % nranksPerNode
  // Step 2: GPU rank i sends data to GPU rank (i+2) % nranksPerNode
  // ...
  // This order is much better for DMA engine for NVLinks
  if (nRanksPerNode == 1) return;

  int startRankInNode = (rank / nRanksPerNode) * nRanksPerNode;
  for (int i = 1; i < nRanksPerNode; i++) {
    int remoteSendToRank = (rank + i) % nRanksPerNode + startRankInNode;
    int remoteRecvFromRank = (rank + nRanksPerNode - i) % nRanksPerNode + startRankInNode;
    int peerSendId = (remoteSendToRank < rank) ? remoteSendToRank : remoteSendToRank - 1;
    int peerRecvId = (remoteRecvFromRank < rank) ? remoteRecvFromRank : remoteRecvFromRank - 1;

    DeviceHandle<mscclpp::PortChannel>& devSendChan = constDevSndRoundChans[peerSendId];
    DeviceHandle<mscclpp::PortChannel>& devRecvChan = constDevSndRoundChans[peerRecvId];
    // wait for the data from GPU (rank-i) % nranksPerNode to arrive
    devSendChan.putWithSignal(offset, size);
    devRecvChan.wait();
  }
}

// Run with a single thread only.
__device__ void allGather(int rank, int worldSize, int nRanksPerNode, size_t nelemsPerGPU) {
  // this allgather is a pipelined and hierarchical one and only works for two nodes
  // it is implemented as follows:
  // Step 1: each node does a local allgather and concurrently,
  // local GPU i exchange (piplineSize-1)/pipelineSize portion of their data with
  // its cross-node neighbor (local GPU i on the other node) via IB
  // Step 2: each node does a local allgather again with the data just received from its
  // cross-node neighbor in step 1, and concurrently, exchange the rest of the data with
  // its cross-node neighbor
  // Step 3: each node does a local allgather for the last time with the rest of the data

  int pipelineSize = 3;
  int peerRank = (rank + nRanksPerNode) % worldSize;
  int peerNodeId = peerRank / nRanksPerNode;
  int peer = (peerRank < rank) ? peerRank : peerRank - 1;
  DeviceHandle<mscclpp::PortChannel>& portChan = constDevSndRoundChans[peer];

  if (peerNodeId == rank / nRanksPerNode) {
    localAllGather(rank, nRanksPerNode, rank * nelemsPerGPU * sizeof(double), nelemsPerGPU * sizeof(double));
    return;
  }

  // Step 1
  portChan.putWithSignal(rank * nelemsPerGPU * sizeof(double),
                         (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(double));
  localAllGather(rank, nRanksPerNode, rank * nelemsPerGPU * sizeof(double), nelemsPerGPU * sizeof(double));
  portChan.wait();
  portChan.flush();
  // Step 2
  portChan.putWithSignal((rank * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(double),
                         nelemsPerGPU / pipelineSize * sizeof(double));
  localAllGather(rank, nRanksPerNode, peerRank * nelemsPerGPU * sizeof(double),
                 (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(double));
  portChan.wait();
  portChan.flush();
  // Step 3
  localAllGather(rank, nRanksPerNode,
                 (peerRank * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(double),
                 nelemsPerGPU / pipelineSize * sizeof(double));
}

__device__ void localReduceScatterMem(double* buff, int rank, int nRanksPerNode, int startChunkIndex, size_t offsetInChunk,
                                      size_t chunkSize, size_t nelems, int nBlocks) {
  if (nRanksPerNode == 1) return;
  if ((int)blockIdx.x >= nBlocks) return;
  const int nPeer = nRanksPerNode - 1;
  DeviceHandle<mscclpp::MemoryChannel>* memChans = constMemOutOfPlaceGetChans;

  const size_t localRankIndexInNode = rank % nRanksPerNode;
  const size_t indexOffset = ((localRankIndexInNode + startChunkIndex) * chunkSize + offsetInChunk);
  
  // Divide by 2 because sizeof(double2) == 2 * sizeof(double)
  const size_t indexOffset2 = indexOffset / 2; 

  // Cast buffer to double2 for vectorized access
  double2* buff2 = (double2*)buff;

  // --- Synchronization (Unchanged) ---
  for (int peerIdx = threadIdx.x + blockIdx.x * blockDim.x; peerIdx < nPeer; peerIdx += blockDim.x * nBlocks) {
    memChans[peerIdx].signal();
  }
  for (int peerIdx = threadIdx.x + blockIdx.x * blockDim.x; peerIdx < nPeer; peerIdx += blockDim.x * nBlocks) {
    memChans[peerIdx].wait();
  }
  reduceScatterDeviceSyncer.sync(nBlocks);
  // -----------------------------------

  // Main Vectorized Loop (Process 2 doubles at a time)
  const size_t nDouble2 = nelems / 2;
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nDouble2; idx += blockDim.x * nBlocks) {
    double2 sum = make_double2(0.0, 0.0);

    for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
      // Read double2 from peer
      double2 val = memChans[peerIdx].read<double2>(indexOffset2 + idx);
      sum.x += val.x;
      sum.y += val.y;
    }
    
    // Accumulate into local buffer
    buff2[indexOffset2 + idx].x += sum.x;
    buff2[indexOffset2 + idx].y += sum.y;
  }

  // Remainder Loop (Process remaining single double if nelems is odd)
  const size_t nLastDoubles = nelems % 2;
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nLastDoubles; idx += blockDim.x * nBlocks) {
    double sum = 0.0;
    for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
      // Read single double
      double val = memChans[peerIdx].read<double>(indexOffset + nDouble2 * 2 + idx);
      sum += val;
    }
    buff[indexOffset + nDouble2 * 2 + idx] += sum;
  }
}

__device__ void localReduceScatterMem2(double* buff, int rank, int nRanksPerNode, size_t chunkSize, size_t nelems,
                                       int nBlocks) {
  if (nRanksPerNode == 1) return;
  if ((int)blockIdx.x >= nBlocks) return;
  const int nPeer = nRanksPerNode - 1;
  DeviceHandle<mscclpp::MemoryChannel>* memChans = constMemOutOfPlaceGetChans;

  const size_t localRankIndexInNode = rank % nRanksPerNode;
  const size_t indexOffset = localRankIndexInNode * chunkSize;
  
  // Changed from /4 to /2 because double2 contains 2 doubles
  const size_t indexOffset2 = indexOffset / 2;

  // Cast to double2 pointer
  double2* buff2 = (double2*)buff;

  // --- Synchronization Logic (Unchanged) ---
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nPeer) {
    memChans[tid].signal();
  }
  const int waitStart = nBlocks * blockDim.x - nPeer;
  if (tid >= waitStart && tid < (int)(nBlocks * blockDim.x)) {
    memChans[tid - waitStart].wait();
  }
  reduceScatterDeviceSyncer.sync(nBlocks);
  // ----------------------------------------

  // Main Loop: Process 2 doubles at a time
  const size_t nDouble2 = nelems / 2;
  for (int index = 0; index < nPeer; ++index) {
    double2 val;
    int peerIdx = (index + localRankIndexInNode) % nPeer;
    
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nDouble2; idx += blockDim.x * nBlocks) {
      // Read double2 from channel
      val = memChans[peerIdx].read<double2>(indexOffset2 + idx);
      
      // Accumulate components (only x and y exist in double2)
      buff2[indexOffset2 + idx].x += val.x;
      buff2[indexOffset2 + idx].y += val.y;
    }
  }

  // Remainder Loop: Process remaining single double if nelems is odd
  const size_t nLastDoubles = nelems % 2;
  for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nLastDoubles; idx += blockDim.x * nBlocks) {
      
      // Calculate correct scalar offset (nDouble2 * 2)
      double val = memChans[(localRankIndexInNode + peerIdx) % nPeer].read<double>(indexOffset + nDouble2 * 2 + idx);
      buff[indexOffset + nDouble2 * 2 + idx] += val;
    }
  }
}

__device__ void localReduceScatterMem3(double* buff, int rank, int nRanksPerNode, size_t chunkSize, size_t nelems,
                                       int nBlocks) {
  if (nRanksPerNode == 1) return;
  if ((int)blockIdx.x >= nBlocks) return;
  const int nPeer = nRanksPerNode - 1;
  DeviceHandle<mscclpp::MemoryChannel>* memChans = constMemOutOfPlaceGetChans;

  const size_t localRankIndexInNode = rank % nRanksPerNode;
  const size_t indexOffset = localRankIndexInNode * chunkSize;
  
  // Divide by 2 (sizeof(double2) / sizeof(double))
  const size_t indexOffset2 = indexOffset / 2;

  // Cast to double2 pointer
  double2* buff2 = (double2*)buff;

  // --- Synchronization (Unchanged) ---
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nPeer) {
    memChans[tid].signal();
  }
  const int waitStart = nBlocks * blockDim.x - nPeer;
  if (tid >= waitStart && tid < (int)(nBlocks * blockDim.x)) {
    memChans[tid - waitStart].wait();
  }
  reduceScatterDeviceSyncer.sync(nBlocks);
  // -----------------------------------

  // Calculate number of double2 elements
  const size_t nDouble2 = nelems / 2;

  size_t base = 0;
  const size_t unitNDouble2 = blockDim.x * nBlocks;

  // 1. Blocked Loop (Optimization for Cache Locality)
  // Iterates through the buffer in chunks of unitNDouble2
  for (; base + unitNDouble2 < nDouble2; base += unitNDouble2) {
    for (int index = 0; index < nPeer; ++index) {
      double2 val;
      int peerIdx = (index + localRankIndexInNode) % nPeer;
      for (size_t idx = base + threadIdx.x + blockIdx.x * blockDim.x; idx < base + unitNDouble2;
           idx += blockDim.x * nBlocks) {
        val = memChans[peerIdx].read<double2>(indexOffset2 + idx);
        buff2[indexOffset2 + idx].x += val.x;
        buff2[indexOffset2 + idx].y += val.y;
      }
    }
  }

  // 2. Cleanup Loop for remaining double2 elements
  for (int index = 0; index < nPeer; ++index) {
    double2 val;
    int peerIdx = (index + localRankIndexInNode) % nPeer;
    for (size_t idx = base + threadIdx.x + blockIdx.x * blockDim.x; idx < nDouble2; idx += blockDim.x * nBlocks) {
      val = memChans[peerIdx].read<double2>(indexOffset2 + idx);
      buff2[indexOffset2 + idx].x += val.x;
      buff2[indexOffset2 + idx].y += val.y;
    }
  }

  // 3. Scalar Remainder Loop (if nelems is odd)
  const size_t nLastDoubles = nelems % 2;
  for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nLastDoubles; idx += blockDim.x * nBlocks) {
      // Calculate scalar offset: nDouble2 * 2
      double val = memChans[(localRankIndexInNode + peerIdx) % nPeer].read<double>(indexOffset + nDouble2 * 2 + idx);
      buff[indexOffset + nDouble2 * 2 + idx] += val;
    }
  }
}

__device__ void reduceScatterMem(double* buff, double* scratch, int rank, int nRanksPerNode, int worldSize,
                                 size_t nelems  // must be divisible by 3
) {
  // this reduce-scatter algorithm works as follows:
  // Step 1: each node does a local reduce-scatter on peer node data chunks with 1/pipeline portion of chunk data. For
  // example, 2 nodes and each node has 2 ranks. rank 0 and rank 1 perform reduce-scatter on chunk 2 and chunk 3, with
  // 1/pipeline portion of the data.
  // Step 2: each node does a local reduce-scatter on peers data chunks with (pipeline-1)/pipeline portion of chunk
  // data. Meanwhile, exchange the reduced data of the previous step with its cross-node neighbor (same local rank
  // number on the other node) via IB. Then performs a reduce operation.
  // Step 3:  each node does a local reduce-scatter on local ranks, meanwhile exchange the reduced data of the previous
  // step with its cross-node neighbor (same local rank number on the other node) via IB. Then performs a reduce
  // operation.
  int pipelineSize = 3;
  float nBlocksForReduceScatterRatio = 0.8;
  const size_t chunkSize = nelems / worldSize;
  const int peerRank = (rank + nRanksPerNode) % worldSize;
  int peerNodeId = peerRank / nRanksPerNode;
  int nBlocksForReduceScatter =
      (int)(nBlocksForReduceScatterRatio * gridDim.x) / (nRanksPerNode - 1) * (nRanksPerNode - 1);
  int isComm = (threadIdx.x == 0) && ((int)blockIdx.x == nBlocksForReduceScatter);
  int peer = (peerRank < rank) ? peerRank : peerRank - 1;
  int nBlocksRemain = gridDim.x - nBlocksForReduceScatter;
  DeviceHandle<mscclpp::PortChannel>& portChan = constDevFstRoundChans[peer];
  if (peerNodeId == rank / nRanksPerNode) {
    localReduceScatterMem(buff, rank, nRanksPerNode, 0, 0, chunkSize, chunkSize, gridDim.x);
    return;
  }

  // step 1: local reduce
  int startChunkIndex = peerNodeId * nRanksPerNode;
  localReduceScatterMem(buff, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize / pipelineSize,
                        nBlocksForReduceScatter);
  deviceSyncer.sync(gridDim.x);

  // step 2: local reduce and exchange data with neighbor
  if (isComm) {
    size_t offset = (peerRank * chunkSize) * sizeof(double);
    // opposite side
    portChan.putWithSignal(offset, (chunkSize / pipelineSize * sizeof(double)));
  }
  localReduceScatterMem(buff, rank, nRanksPerNode, startChunkIndex, chunkSize / pipelineSize, chunkSize,
                        2 * chunkSize / pipelineSize, nBlocksForReduceScatter);
  if (isComm) {
    portChan.wait();
  }
  if ((int)blockIdx.x >= nBlocksForReduceScatter) {
    ibDeviceSyncer.sync(nBlocksRemain);
    // reduce data received from peer to related rank
    size_t offset = rank * chunkSize * sizeof(double);
    double* dst = (double*)((char*)buff + offset);
    double* src = (double*)((char*)scratch + offset);
    vectorSum(dst, src, chunkSize / pipelineSize, blockIdx.x - nBlocksForReduceScatter, nBlocksRemain);
  }
  if (isComm) {
    portChan.flush();
  }
  deviceSyncer.sync(gridDim.x);

  // step 3: local reduce and exchange data with neighbor
  startChunkIndex = (rank / nRanksPerNode) * nRanksPerNode;
  if (isComm) {
    size_t offset = (peerRank * chunkSize + chunkSize / pipelineSize) * sizeof(double);
    portChan.putWithSignal(offset, (pipelineSize - 1) * chunkSize / pipelineSize * sizeof(double));
  }
  localReduceScatterMem(buff, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize, nBlocksForReduceScatter);
  if (isComm) {
    portChan.wait();
  }
  deviceSyncer.sync(gridDim.x);
  // reduce to related rank, can not overlap since localReduceScatter also calculate the sum
  size_t offset = (rank * chunkSize + chunkSize / pipelineSize) * sizeof(double);
  double* dst = (double*)((char*)buff + offset);
  double* src = (double*)((char*)scratch + offset);
  vectorSum(dst, src, 2 * chunkSize / pipelineSize);
  if (isComm) {
    portChan.flush();
  }
}

// This kernel is the most performant when the number of blocks is a multiple of (nRanksPerNode - 1).
__device__ void localAllGatherMem(int rank, int nRanksPerNode, int startRankChunkIndex, uint64_t offsetInRankChunk,
                                  uint64_t rankChunkSize, uint64_t size, size_t nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t peerIdx = blockIdx.x % nPeer;
  const size_t nBlockForThisPeer = nBlocks / nPeer + (nBlocks % nPeer > peerIdx ? 1 : 0);
  const size_t peerLocalBlockIdx = blockIdx.x / nPeer;
  const size_t rankLocalIndex = rank % nRanksPerNode;
  const int remoteRankLocalIndex = (peerIdx < rankLocalIndex ? peerIdx : peerIdx + 1);

  // Split the data into chunks for aligned data access. Ignore the remainder here and let the last block handle it.
  constexpr size_t chunkBytes = 128;  // heuristic value
  const size_t nChunk = size / chunkBytes;
  const size_t nMinChunkPerBlock = nChunk / nBlockForThisPeer;
  const size_t nRemainderChunk = nChunk % nBlockForThisPeer;

  // Distribute chunks to blocks
  size_t nChunkForThisBlock;
  size_t offsetForThisBlock;
  if (peerLocalBlockIdx < nRemainderChunk) {
    nChunkForThisBlock = nMinChunkPerBlock + 1;
    offsetForThisBlock = (nMinChunkPerBlock + 1) * peerLocalBlockIdx;
  } else {
    nChunkForThisBlock = nMinChunkPerBlock;
    offsetForThisBlock =
        (nMinChunkPerBlock + 1) * nRemainderChunk + (peerLocalBlockIdx - nRemainderChunk) * nMinChunkPerBlock;
  }
  offsetForThisBlock *= chunkBytes;

  // Calculate the size of the data for this block
  size_t sizeForThisBlock = nChunkForThisBlock * chunkBytes;
  const size_t lastChunkSize = size - nChunk * chunkBytes;
  if (lastChunkSize > 0 && peerLocalBlockIdx == nBlockForThisPeer - 1) {
    sizeForThisBlock += lastChunkSize;
  }
  if (threadIdx.x == 0 && peerLocalBlockIdx == 0) {
    constMemInPlaceChans[peerIdx].signal();
    constMemInPlaceChans[peerIdx].wait();
  }
  allGatherDeviceSyncer.sync(nBlocks);
  size_t offset = rankChunkSize * (startRankChunkIndex + remoteRankLocalIndex) + offsetInRankChunk;
  constMemInPlaceChans[peerIdx].get(offset + offsetForThisBlock, sizeForThisBlock, threadIdx.x, blockDim.x);
}

__device__ void localRingAllGatherMem(int rank, int nRanksPerNode, uint64_t size, size_t nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nPeer = nRanksPerNode - 1;

  if (tid < nPeer) {
    constMemInPlaceChans[tid].signal();
  }
  int waitStart = nBlocks * blockDim.x - nPeer;
  if (tid >= waitStart && tid < (int)(nBlocks * blockDim.x)) {
    constMemInPlaceChans[tid - waitStart].wait();
  }
  allGatherDeviceSyncer.sync(nBlocks);
  for (int i = 0; i < nPeer; ++i) {
    int peerIdx = (i + rank) % nPeer;
    const int remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    size_t offset = size * remoteRankLocalIndex;
    constMemInPlaceChans[peerIdx].get(offset, size, tid, blockDim.x * nBlocks);
  }
}

__device__ void localRingAllGatherMem2(size_t rank, size_t nRanksPerNode, size_t size, size_t nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;

  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nPeer = nRanksPerNode - 1;

  if (tid < nPeer) {
    constMemInPlaceChans[tid].signal();
  }
  size_t waitStart = nBlocks * blockDim.x - nPeer;
  if (tid >= waitStart && tid < nBlocks * blockDim.x) {
    constMemInPlaceChans[tid - waitStart].wait();
  }
  allGatherDeviceSyncer.sync(nBlocks);
  const size_t unitSize = 16 * blockDim.x * nBlocks;
  size_t base = 0;
  for (; base + unitSize < size; base += unitSize) {
    for (size_t i = 0; i < nPeer; ++i) {
      size_t peerIdx = (i + rank) % nPeer;
      const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
      size_t offset = size * remoteRankLocalIndex + base;
      constMemInPlaceChans[peerIdx].get(offset, unitSize, tid, blockDim.x * nBlocks);
    }
  }
  for (size_t i = 0; i < nPeer; ++i) {
    size_t peerIdx = (i + rank) % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    size_t offset = size * remoteRankLocalIndex + base;
    constMemInPlaceChans[peerIdx].get(offset, size - base, tid, blockDim.x * nBlocks);
  }
}

// This is an allgather4 equivalent
__device__ void allGatherMem(int rank, int worldSize, int nRanksPerNode, size_t nelemsPerGPU) {
  // this allgather is a pipelined and hierarchical one and only works for two nodes
  // it is implemented as follows:
  // Step 1: each node does a local allgather and concurrently,
  // local GPU i exchange (piplineSize-1)/pipelineSize portion of their data with
  // its cross-node neighbor (local GPU i on the other node) via IB
  // Step 2: each node does a local allgather again with the data just received from its
  // cross-node neighbor in step 1, and concurrently, exchange the rest of the data with
  // its cross-node neighbor
  // Step 3: each node does a local allgather for the last time with the rest of the data

  int pipelineSize = 3;
  int peerRank = (rank + nRanksPerNode) % worldSize;
  int peerNodeId = peerRank / nRanksPerNode;
  int peer = (peerRank < rank) ? peerRank : peerRank - 1;
  DeviceHandle<mscclpp::PortChannel>& portChan = constDevSndRoundChans[peer];
  const size_t nBlocksForLocalAllGather = gridDim.x / (nRanksPerNode - 1) * (nRanksPerNode - 1);
  const size_t rankChunkSize = nelemsPerGPU * sizeof(double);
  const int startRankIndexInLocalNode = (rank / nRanksPerNode) * nRanksPerNode;
  const int startRankIndexInPeerNode = (peerRank / nRanksPerNode) * nRanksPerNode;

  if (peerNodeId == rank / nRanksPerNode) {
    localAllGatherMem(rank, nRanksPerNode, 0, 0, rankChunkSize, rankChunkSize, gridDim.x);
    return;
  }

  constexpr size_t alignment = 128;
  size_t step1Bytes = (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(double);
  step1Bytes = step1Bytes / alignment * alignment;
  const size_t step2Bytes = nelemsPerGPU * sizeof(double) - step1Bytes;

  // Step 1
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    portChan.putWithSignal(rank * nelemsPerGPU * sizeof(double), step1Bytes);
  }
  localAllGatherMem(rank, nRanksPerNode, startRankIndexInLocalNode, 0, rankChunkSize, rankChunkSize,
                    nBlocksForLocalAllGather);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    portChan.wait();
    portChan.flush();
  }
  deviceSyncer.sync(gridDim.x);
  // Step 2
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    portChan.putWithSignal(rank * nelemsPerGPU * sizeof(double) + step1Bytes, step2Bytes);
  }
  localAllGatherMem(rank, nRanksPerNode, startRankIndexInPeerNode, 0, rankChunkSize, step1Bytes,
                    nBlocksForLocalAllGather);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    portChan.wait();
    portChan.flush();
  }
  deviceSyncer.sync(gridDim.x);
  // Step 3
  localAllGatherMem(rank, nRanksPerNode, startRankIndexInPeerNode, step1Bytes, rankChunkSize, step2Bytes,
                    nBlocksForLocalAllGather);
}


__global__ void __launch_bounds__(1024)
    allreduce0(double* buff, double* scratch, int rank, int worldSize, size_t nelems, size_t scratchDataCount) {
  int peerId = blockIdx.x / BLOCKS_PER_PEER;
  int isComm = (threadIdx.x == 0) && (blockIdx.x % BLOCKS_PER_PEER == 0);
  int remoteRank = (peerId < rank) ? peerId : peerId + 1;

  // 1st communication phase: send data to the scratch buffer of the peer associated with this block
  DeviceHandle<mscclpp::PortChannel>& devFstRoundChan = constDevFstRoundChans[peerId];
  Chunk toPeerChunk = getChunk(nelems, worldSize, remoteRank);
  // Now we need to figure out the offset of this chunk in the scratch buffer of the destination.
  // The destination will have allocated a scratch buffer of size numPeers() * toPeerChunk.size and
  // inside that each of the destination's peers send to the nth chunk, where n is the index of the
  // source peer from the destination's perspective.
  size_t dstOffset = (rank < remoteRank ? rank : rank - 1) * toPeerChunk.size;
  if (isComm) {
    // Write data to the peer
    // Changed sizeof(int) to sizeof(double)
    devFstRoundChan.putWithSignalAndFlush(dstOffset * sizeof(double), toPeerChunk.offset * sizeof(double),
                                          toPeerChunk.size * sizeof(double));
    // Wait for data from the peer
    devFstRoundChan.wait();
  }

  deviceSyncer.sync(gridDim.x);

  // Local reduction: every block reduces a slice of each chunk in the scratch buffer into the user buffer
  DeviceHandle<mscclpp::PortChannel>& devSndRoundChan = constDevSndRoundChans[peerId];
  Chunk rankChunk = getChunk(nelems, worldSize, rank);
  
  // Changed int* to double*
  double* chunk = buff + rankChunk.offset;
  
  int numPeers = gridDim.x / BLOCKS_PER_PEER;
  int numBlocks = gridDim.x;
  Chunk blockUserChunk = getChunk(rankChunk.size, numBlocks, blockIdx.x);
  size_t scratchDataCountPerPeer = scratchDataCount / numPeers;
  Chunk blockScratchChunk = getChunk(scratchDataCountPerPeer, numBlocks, blockIdx.x);
  for (int peerIdx = 0; peerIdx < numPeers; ++peerIdx) {
    // Changed int* to double*
    double* scratchChunk = scratch + peerIdx * scratchDataCountPerPeer;
    
    // Ensure vectorSumSingleBlock is capable of handling double*
    vectorSumSingleBlock(chunk + blockUserChunk.offset, scratchChunk + blockScratchChunk.offset,
                         blockScratchChunk.size);
  }

  deviceSyncer.sync(gridDim.x);

  // 2nd communication phase: send the now reduced data between the user buffers
  Chunk collectionChunk = getChunk(nelems, worldSize, rank);
  if (isComm) {
    // Write data to the peer
    // Changed sizeof(int) to sizeof(double)
    devSndRoundChan.putWithSignalAndFlush(collectionChunk.offset * sizeof(double), collectionChunk.offset * sizeof(double),
                                          collectionChunk.size * sizeof(double));
    // Wait for data from the peer
    devSndRoundChan.wait();
  }
}

__global__ void __launch_bounds__(1024) allreduce1(double* buff, double* scratch, int rank, int worldSize, size_t nelems) {
  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int remoteSendRank = (rank + 1) % worldSize;
  int remoteRecvRank = (rank + worldSize - 1) % worldSize;
  int peerSendId = (remoteSendRank < rank) ? remoteSendRank : remoteSendRank - 1;
  int peerRecvId = (remoteRecvRank < rank) ? remoteRecvRank : remoteRecvRank - 1;

  DeviceHandle<mscclpp::PortChannel>& devFstSendChan = constDevFstRoundChans[peerSendId];
  DeviceHandle<mscclpp::PortChannel>& devFstRecvChan = constDevFstRoundChans[peerRecvId];
  DeviceHandle<mscclpp::PortChannel>& devSndSendChan = constDevSndRoundChans[peerSendId];
  DeviceHandle<mscclpp::PortChannel>& devSndRecvChan = constDevSndRoundChans[peerRecvId];

  // Step 1
  size_t chunkIndex = (rank + worldSize - 1) % worldSize;
  size_t chunkNelem = nelems / worldSize;
  // CHANGED: sizeof(int) -> sizeof(double)
  size_t offset = chunkIndex * chunkNelem * sizeof(double); 

  if (isComm) {
    if (chunkNelem > 1) {
       // CHANGED: sizeof(int) -> sizeof(double)
      devFstSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(double));
    }
  }

  // Step 2 ~ Step n-1
  for (int step = 2; step < worldSize; ++step) {
    if (isComm) {
      if (chunkNelem > 1) {
        devFstRecvChan.wait();
        devFstSendChan.flush();
      }
      // CHANGED: sizeof(int) -> sizeof(double)
      devFstSendChan.putWithSignal(offset + chunkNelem / 2 * sizeof(double), (chunkNelem - chunkNelem / 2) * sizeof(double));
    }
    deviceSyncer.sync(gridDim.x);

    // Reduce
    chunkIndex = (rank + worldSize - step) % worldSize;
    // CHANGED: sizeof(int) -> sizeof(double)
    offset = chunkIndex * chunkNelem * sizeof(double);
    
    // CHANGED: int* -> double*
    double* dst = (double*)((char*)buff + offset);
    double* src = (double*)((char*)scratch + offset);
    
    // Note: Ensure vectorSum is overloaded or updated to handle double* // and uses double2 for vectorized loads.
    vectorSum(dst, src, chunkNelem / 2);

    if (isComm) {
      devFstRecvChan.wait();
      devFstSendChan.flush();
      if (chunkNelem > 1) {
        // CHANGED: sizeof(int) -> sizeof(double)
        devFstSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(double));
      }
    }
    deviceSyncer.sync(gridDim.x);

    dst += chunkNelem / 2;
    src += chunkNelem / 2;
    vectorSum(dst, src, chunkNelem - chunkNelem / 2);
  }

  // Step n
  if (isComm) {
    if (chunkNelem > 1) {
      devFstRecvChan.wait();
      devFstSendChan.flush();
    }
    // CHANGED: sizeof(int) -> sizeof(double)
    devFstSendChan.putWithSignal(offset + chunkNelem / 2 * sizeof(double), (chunkNelem - chunkNelem / 2) * sizeof(double));
  }
  deviceSyncer.sync(gridDim.x);

  // CHANGED: sizeof(int) -> sizeof(double)
  offset = rank * chunkNelem * sizeof(double);
  
  // CHANGED: int* -> double*
  double* dst = (double*)((char*)buff + offset);
  double* src = (double*)((char*)scratch + offset);
  vectorSum(dst, src, chunkNelem / 2);

  if (isComm) {
    devFstRecvChan.wait();
    devFstSendChan.flush();
    if (chunkNelem > 1) {
      // CHANGED: sizeof(int) -> sizeof(double)
      devSndSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(double));
    }
  }
  deviceSyncer.sync(gridDim.x);

  dst += chunkNelem / 2;
  src += chunkNelem / 2;
  vectorSum(dst, src, chunkNelem - chunkNelem / 2);

  if (isComm) {
    if (chunkNelem > 1) {
      devSndRecvChan.wait();
      devSndSendChan.flush();
    }
    // CHANGED: sizeof(int) -> sizeof(double)
    devSndSendChan.putWithSignalAndFlush(offset + chunkNelem / 2 * sizeof(double),
                                         (chunkNelem - chunkNelem / 2) * sizeof(double));
  }

  // Step n+1 ~ Step 2n-2
  for (int i = 1; i < worldSize - 1; ++i) {
    if (isComm) {
      devSndRecvChan.wait();
    }
    deviceSyncer.sync(gridDim.x);

    // Copy
    chunkIndex = (rank + worldSize - i) % worldSize;
    if (isComm) {
      // CHANGED: sizeof(int) -> sizeof(double)
      devSndSendChan.putWithSignalAndFlush(chunkIndex * chunkNelem * sizeof(double), chunkNelem * sizeof(double));
    }
  }

  // Final receive
  if (isComm) {
    devSndRecvChan.wait();
  }
}

// it is now sending 1 double (64 bit) rather than 2 int (2 x 32 bit).
// to achieve that using LLPacket: for the road, cast them as long long int so that 64 bit double acts as 2 int.
// when you reach the destination: back from long long to double for computation.
__global__ void __launch_bounds__(1024)
    allreduce2(double* buff, void* scratch, void* putPktBuf, void* getPktBuf, void* result, int rank, int nRanksPerNode,
               int worldSize, size_t nelems) {
  int numPeersPerNode = nRanksPerNode - 1;
  size_t nPkts = nelems;  // 1 double per packet
  size_t pktBytes = nPkts * sizeof(mscclpp::LLPacket);

  // Channel to a local peer
  int memChanIdx = blockIdx.x / BLOCKS_PER_PEER;
  DeviceHandle<mscclpp::MemoryChannel> memChan = constMemOutOfPlaceChans[memChanIdx];

  // Channel to a remote peer that has the same local rank as me
  int localRank = rank % nRanksPerNode;
  DeviceHandle<mscclpp::PortChannel> portChan = constDevFstRoundChans[localRank];

  // Flag for packets. Initially 1
  uint32_t flag = (uint32_t)globalFlag;

  double* src = buff;
  double* res = (double*)result;
  // double buffering
  size_t scratchBaseIndex = (flag & 1) ? 0 : nPkts * max(numPeersPerNode, 1);
  size_t scratchOffset = scratchBaseIndex * sizeof(mscclpp::LLPacket);
  size_t pktBufOffset = (flag & 1) ? 0 : nPkts * sizeof(mscclpp::LLPacket);
  mscclpp::LLPacket* getPktPtr = (mscclpp::LLPacket*)((char*)getPktBuf + pktBufOffset);
  mscclpp::LLPacket* putPktPtr = (mscclpp::LLPacket*)((char*)putPktBuf + pktBufOffset);

  // Phase 1: Local AllReduce. Read from buff, write to putPktBuf (for single node) or to result (for 2 nodes)
  if (numPeersPerNode == 0) {
    // One rank per node: write data to putPktBuf directly
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * gridDim.x) {
      uint64_t data = __double_as_longlong(src[idx]);
      // mask with 0xFFFFFFFF: the lower 32 bits, shift by 32: higher 32 bits
      putPktPtr[idx].write((uint32_t)(data & 0xFFFFFFFF), (uint32_t)(data >> 32), flag);
    }
  } else {
    // Offset of the input data (buff) to read from
    size_t srcOffset =
        ((blockIdx.x % BLOCKS_PER_PEER) * nelems * sizeof(double) / BLOCKS_PER_PEER);  // offset for this block
    // Offset of the peer's scratch buffer (scratch) to write on
    size_t dstOffset = (scratchOffset) +                                                    // double buffering
                       ((memChanIdx < localRank ? localRank - 1 : localRank) * pktBytes) +  // offset for this rank
                       (srcOffset);  // offset for this block
    // Write data to the peer's scratch
    memChan.putPackets(dstOffset, srcOffset, nelems / BLOCKS_PER_PEER * sizeof(double), threadIdx.x, blockDim.x, flag);
    // Read data from my scratch, reduce data with my buff, and write the result to my putPktBuf or to result
    const bool isSingleNode = (worldSize == nRanksPerNode);
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * gridDim.x) {
      double sum = 0.0;
      for (int peerIdx = 0; peerIdx < numPeersPerNode / 2; ++peerIdx) {
        uint2 data0 = memChan.unpackPacket(scratchBaseIndex + 2 * peerIdx * nPkts + idx, flag); // even peers
        uint2 data1 = memChan.unpackPacket(scratchBaseIndex + (2 * peerIdx + 1) * nPkts + idx, flag); // odd peers
        // data0.y: Holds the Upper 32 bits, data0.x: Holds the Lower 32 bits
        sum += __longlong_as_double(((uint64_t)data0.y << 32) | data0.x);
        sum += __longlong_as_double(((uint64_t)data1.y << 32) | data1.x);
      }
      if (numPeersPerNode & 1) {
        uint2 data = memChan.unpackPacket(scratchBaseIndex + (numPeersPerNode - 1) * nPkts + idx, flag); // very last peer
        sum += __longlong_as_double(((uint64_t)data.y << 32) | data.x);
      }
      double total = src[idx] + sum;
      if (isSingleNode) { // finalize: my local + accumulated sum
        res[idx] = total;
      } else {
        uint64_t totalBits = __double_as_longlong(total);
        putPktPtr[idx].write((uint32_t)(totalBits & 0xFFFFFFFF), (uint32_t)(totalBits >> 32), flag); // send to the other node in outbox
      }
    }
  }

  // If this is single node AllReduce, we are done.
  if (worldSize != nRanksPerNode) {
    // Phase 2: Inter-node AllReduce. Supports only 2 nodes. Read from putPktBuf, write to result

    // Wait for all threads to finish writing to putPktBuf in Phase 1
    deviceSyncer.sync(gridDim.x);

    // Phase 2 may need less blocks than Phase 1.
    constexpr int nBlocksPhase2 = 1;
    if (blockIdx.x >= nBlocksPhase2) return;

    // Write my putPktBuf to the remote peer's getPktBuf
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      portChan.put(pktBufOffset, pktBytes);
      if ((flag & 63) == 0) {
        portChan.flush();
      }
    }

    // Read data from my getPktBuf, reduce data with my putPktBuf, and write the result to result
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * nBlocksPhase2) {
      uint2 data0 = putPktPtr[idx].read(flag);
      uint2 data1 = getPktPtr[idx].read(flag);
      double val0 = __longlong_as_double(((uint64_t)data0.y << 32) | data0.x);
      double val1 = __longlong_as_double(((uint64_t)data1.y << 32) | data1.x);
      res[idx] = val0 + val1;
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

__global__ void __launch_bounds__(1024)
    allreduce3(double* buff, double* scratch, int rank, int nRanksPerNode, int worldSize, size_t nelems) {
  reduceScatter(buff, scratch, rank, nRanksPerNode, worldSize, nelems);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    allGather(rank, worldSize, nRanksPerNode, nelems / worldSize);
  }
}

__global__ void __launch_bounds__(1024)
    allreduce4(double* buff, double* scratch, int rank, int nRanksPerNode, int worldSize, size_t nelems) {
  reduceScatterMem(buff, scratch, rank, nRanksPerNode, worldSize, nelems);
  deviceSyncer.sync(gridDim.x);
  allGatherMem(rank, worldSize, nRanksPerNode, nelems / worldSize);
}

__global__ void __launch_bounds__(1024)
    allreduce5(double* buff, double rank, int nRanksPerNode, int worldSize, size_t nelems) {
#if defined(__HIP_PLATFORM_AMD__)
  localReduceScatterMem3(buff, rank, nRanksPerNode, nelems / worldSize, nelems / worldSize, gridDim.x);
  deviceSyncer.sync(gridDim.x);
  localRingAllGatherMem2(rank, nRanksPerNode, nelems / worldSize * sizeof(double), gridDim.x);
#else
  localReduceScatterMem2(buff, rank, nRanksPerNode, nelems / worldSize, nelems / worldSize, gridDim.x);
  deviceSyncer.sync(gridDim.x);
  localRingAllGatherMem(rank, nRanksPerNode, nelems / worldSize * sizeof(double), gridDim.x);
#endif
}

__global__ void __launch_bounds__(1024)
    allreduce6(double* buff, double* scratch, void* resultBuff, int rank, int nRanksPerNode, int worldSize, size_t nelems) {
  // This version of allreduce only works for single nodes
  const int nPeers = nRanksPerNode - 1;
  const size_t nPkts = nelems / 2; // 2 doubles per packet (128-bit)
  const int nelemsPerRank = nelems / worldSize;
  const int nPktsPerRank = nelemsPerRank / 2;
  
  // flag for packets. Initially 1
  const uint32_t flag = (uint32_t)globalFlag;
  
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  
  // double buffering
  size_t scratchBaseIndex = (flag & 1) ? 0 : nPkts;
  size_t scratchBaseOffset = scratchBaseIndex * sizeof(mscclpp::LLPacket);
  size_t scratchOffset = scratchBaseOffset + rank * nPktsPerRank * sizeof(mscclpp::LLPacket);
  size_t scratchResultIndex = (flag & 1) ? 2 * nPkts : 3 * nPkts;
  
  // CHANGED: sizeof(int) -> sizeof(double)
  size_t srcOffset = remoteRank * nelemsPerRank * sizeof(double);
  
  // CHANGED: uint2* (64-bit load) -> double2* (128-bit load)
  // CHANGED: sizeof(int) -> sizeof(double)
  double2* src = (double2*)((char*)buff + rank * nelemsPerRank * sizeof(double));
  double2* dst = (double2*)((char*)resultBuff + rank * nelemsPerRank * sizeof(double));

  // step 1: write to scratch buffer
  // CHANGED: sizeof(int) -> sizeof(double)
  constMemOutOfPlaceChans[peerIdx].putPackets(scratchOffset, srcOffset, nelemsPerRank * sizeof(double), tid,
                                              blockDim.x * nBlocksPerPeer, flag);
                                              
  // step 2: get data from scratch buffer, reduce data and write result to remote scratch buffer
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * gridDim.x) {
    // CHANGED: uint2 -> double2 accumulation
    double2 data;
    data.x = 0.0; 
    data.y = 0.0;

    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      
      // Assumption: unpackPacket returns a type with 64-bit .x and .y fields (like ulong2)
      auto val = constMemOutOfPlaceChans[peerIdx].unpackPacket(scratchBaseIndex + remoteRank * nPktsPerRank + idx, flag);
      
      // CHANGED: Reinterpret bits from packet (uint64/longlong) to double
      data.x += __longlong_as_double(val.x);
      data.y += __longlong_as_double(val.y);
    }
    
    data.x += src[idx].x;
    data.y += src[idx].y;
    dst[idx] = data;

    mscclpp::LLPacket packet;
    // CHANGED: Reinterpret double bits to longlong for storage in packet
    packet.data1 = __double_as_longlong(data.x);
    packet.flag1 = flag;
    packet.data2 = __double_as_longlong(data.y);
    packet.flag2 = flag;
    
    size_t offset = scratchResultIndex + (idx + rank * nPktsPerRank);
    for (int index = 0; index < nPeers; index++) {
      constMemOutOfPlaceChans[index].write(offset, packet);
    }
  }

  // step 3: get data result from scratch buffer
  const int dstOffset = remoteRank * nPktsPerRank;
  // CHANGED: uint2* -> double2*
  // CHANGED: sizeof(int) -> sizeof(double)
  double2* result = (double2*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(double));
  
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * nBlocksPerPeer) {
    auto data = constMemOutOfPlaceChans[peerIdx].unpackPacket(scratchResultIndex + dstOffset + idx, flag);
    
    // CHANGED: Reinterpret bits back to double
    result[idx].x = __longlong_as_double(data.x);
    result[idx].y = __longlong_as_double(data.y);
  }
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

// __global__ void __launch_bounds__(1024)
//     allreduce7(double* buff, double* scratch, void* resultBuff, int rank, int nRanksPerNode, int worldSize, size_t nelems) {
//   // This version of allreduce only works for single nodes
//   const int nPeers = nRanksPerNode - 1;
  
//   // CHANGED: We are processing 2 doubles per packet (16 bytes)
//   const size_t nPkts = nelems / 2; 
  
//   const int nelemsPerRank = nelems / worldSize;
//   const int nPktsPerRank = nelemsPerRank / 2;

//   // flag for packets. Initially 1
//   const uint32_t flag = (uint32_t)globalFlag;
  
//   // thread block & channel info
//   const int nBlocksPerPeer = gridDim.x / nPeers;
//   const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
//   const int peerIdx = blockIdx.x / nBlocksPerPeer;
//   const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
//   const int tid = threadIdx.x + localBlockIdx * blockDim.x;

//   // double buffering
//   // CHANGED: LL8Packet -> LLPacket (16 bytes data)
//   size_t scratchBaseIndex = (flag & 1) ? 0 : nPkts;
//   size_t scratchOffset = (scratchBaseIndex + rank * nPktsPerRank) * sizeof(mscclpp::LLPacket);
//   size_t scratchResultIndex = (flag & 1) ? 2 * nPkts : 3 * nPkts;
  
//   // CHANGED: sizeof(int) -> sizeof(double)
//   size_t srcOffset = remoteRank * nelemsPerRank * sizeof(double);
  
//   // CHANGED: uint32_t* -> double2* (loading 2 doubles / 128 bits)
//   double2* src = (double2*)((char*)buff + rank * nelemsPerRank * sizeof(double));
//   double2* dst = (double2*)((char*)resultBuff + rank * nelemsPerRank * sizeof(double));

//   // step 1: write to scratch buffer
//   // CHANGED: LL8Packet -> LLPacket, sizeof(int) -> sizeof(double)
//   constMemOutOfPlaceChans[peerIdx].putPackets<mscclpp::LLPacket>(scratchOffset, srcOffset, nelemsPerRank * sizeof(double),
//                                                                   tid, blockDim.x * nBlocksPerPeer, flag);
                                                                  
//   // step 2: get data from scratch buffer, reduce data and write result to remote scratch buffer
//   for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * gridDim.x) {
//     // CHANGED: Accumulator is now double2
//     double2 data; 
//     data.x = 0.0; 
//     data.y = 0.0;

//     for (int index = 0; index < nPeers; index++) {
//       const int remoteRank = index < rank ? index : index + 1;
      
//       // CHANGED: Unpack LLPacket (expecting 128-bit payload split into two 64-bit integers)
//       auto val = constMemOutOfPlaceChans[peerIdx].unpackPacket<mscclpp::LLPacket>(
//           scratchBaseIndex + remoteRank * nPktsPerRank + idx, flag);
      
//       // CHANGED: Bit-cast from transport integer type to double for math
//       data.x += __longlong_as_double(val.data1); // assuming LLPacket has .data1 / .data2
//       data.y += __longlong_as_double(val.data2); 
//     }
    
//     data.x += src[idx].x;
//     data.y += src[idx].y;
//     dst[idx] = data;

//     // CHANGED: Prepare LLPacket
//     mscclpp::LLPacket packet;
//     // CHANGED: Bit-cast double result back to integer for transport
//     packet.data1 = __double_as_longlong(data.x);
//     packet.flag1 = flag;
//     packet.data2 = __double_as_longlong(data.y);
//     packet.flag2 = flag;
    
//     size_t offset = scratchResultIndex + (idx + rank * nPktsPerRank);
//     for (int index = 0; index < nPeers; index++) {
//       constMemOutOfPlaceChans[index].write(offset, packet);
//     }
//   }

//   // step 3: get data result from scratch buffer
//   const int dstOffset = remoteRank * nPktsPerRank;
//   // CHANGED: uint32_t* -> double2*
//   double2* result = (double2*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(double));
  
//   for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * nBlocksPerPeer) {
//     // CHANGED: Unpack LLPacket
//     auto data =
//         constMemOutOfPlaceChans[peerIdx].unpackPacket<mscclpp::LLPacket>(scratchResultIndex + dstOffset + idx, flag);
    
//     // CHANGED: Bit-cast back to double
//     result[idx].x = __longlong_as_double(data.data1);
//     result[idx].y = __longlong_as_double(data.data2);
//   }

//   if (threadIdx.x == 0 && blockIdx.x == 0) {
//     globalFlag += 1;
//   }
// }

class AllReduceTestColl : public BaseTestColl {
 public:
  AllReduceTestColl() = default;
  ~AllReduceTestColl() = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) override;
  void getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size) override;
  std::vector<KernelRestriction> getKernelRestrictions() override;
};

void AllReduceTestColl::runColl(const TestArgs& args, cudaStream_t stream) {
  const int worldSize = args.totalRanks;
  const int rank = args.rank;
  const int kernelNum = args.kernelNum;
  const int nPeers = worldSize - 1;
  const Chunk chunk = getChunk(paramCount_, worldSize, rank);
  const size_t scratchDataCount = chunk.size * nPeers;

  int nBlocks;
  int nThreadsPerBlock;
  void* tmpBuff;
  if (kernelNum == 0) {
    nBlocks = nPeers * BLOCKS_PER_PEER;
    tmpBuff = scratchBuff;
    nThreadsPerBlock = 1024;
  } else if (kernelNum == 1 || kernelNum == 3) {
    nBlocks = 24;
    tmpBuff = scratchBuff;
    nThreadsPerBlock = 1024;
  } else if (kernelNum == 4) {
    nBlocks = 45;
    tmpBuff = scratchBuff;
    nThreadsPerBlock = 512;
  } else if (kernelNum == 5) {
    nBlocks = 24;
    tmpBuff = scratchBuff;
    nThreadsPerBlock = 1024;
  } else if (kernelNum == 6) {
    nBlocks = 21;
    tmpBuff = scratchPacketBuff;
    nThreadsPerBlock = 512;
  } else if (kernelNum == 7) {
    nBlocks = 28;
    tmpBuff = scratchPacketBuff;
    nThreadsPerBlock = 1024;
  } else {
    nBlocks = std::max(args.nRanksPerNode - 1, 1) * BLOCKS_PER_PEER;
    tmpBuff = scratchPacketBuff;
    nThreadsPerBlock = 1024;
  }
  if (kernelNum == 0)
    allreduce0<<<nBlocks, nThreadsPerBlock, 0, stream>>>((double*)inputBuff, (double*)tmpBuff, rank, worldSize, paramCount_,
                                                         scratchDataCount);
  else if (kernelNum == 1)
    allreduce1<<<nBlocks, nThreadsPerBlock, 0, stream>>>((double*)inputBuff, (double*)tmpBuff, rank, worldSize, paramCount_);
  else if (kernelNum == 2)
    allreduce2<<<nBlocks, nThreadsPerBlock, 0, stream>>>((double*)inputBuff, tmpBuff, putPacketBuff, getPacketBuff,
                                                         resultBuff, rank, args.nRanksPerNode, worldSize, paramCount_);
  else if (kernelNum == 3)
    allreduce3<<<nBlocks, nThreadsPerBlock, 0, stream>>>((double*)inputBuff, (double*)tmpBuff, rank, args.nRanksPerNode,
                                                         worldSize, paramCount_);
  else if (kernelNum == 4)
    allreduce4<<<nBlocks, nThreadsPerBlock, 0, stream>>>((double*)inputBuff, (double*)tmpBuff, rank, args.nRanksPerNode,
                                                         worldSize, paramCount_);
  else if (kernelNum == 5)
    allreduce5<<<nBlocks, nThreadsPerBlock, 0, stream>>>((double*)inputBuff, rank, args.nRanksPerNode, worldSize,
                                                         paramCount_);
  else if (kernelNum == 6)
    allreduce6<<<nBlocks, nThreadsPerBlock, 0, stream>>>((double*)inputBuff, (double*)tmpBuff, resultBuff, rank,
                                                         args.nRanksPerNode, worldSize, paramCount_);
  // else if (kernelNum == 7)
  //   allreduce7<<<nBlocks, nThreadsPerBlock, 0, stream>>>((double*)inputBuff, (double*)tmpBuff, resultBuff, rank,
  //                                                        args.nRanksPerNode, worldSize, paramCount_);
}

void AllReduceTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  if (sendBuff.size() != 1) std::runtime_error("unexpected error");
  const int rank = args.rank;
  const int worldSize = args.totalRanks;
  std::vector<double> dataHost(std::max(sendCount_, recvCount_), rank);
  CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), sendCount_ * typeSize_, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = worldSize * (worldSize - 1) / 2;
  }
  std::memcpy(expectedBuff, dataHost.data(), recvCount_ * typeSize_);
}

void AllReduceTestColl::getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) {
  double baseBw = (double)(paramCount_ * typeSize_) / 1.0E9 / deltaSec;
  algBw = baseBw;
  double factor = (2 * (double)(worldSize_ - 1)) / ((double)worldSize_);
  busBw = baseBw * factor;
}

void AllReduceTestColl::setupCollTest(size_t size) {
  size_t count = size / typeSize_;
  sendCount_ = count;
  recvCount_ = count;
  paramCount_ = count;
  expectedCount_ = count;

  mscclpp::DeviceSyncer syncer = {};
  uint64_t initFlag = 1;
  CUDATHROW(cudaMemcpyToSymbol(deviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
  CUDATHROW(cudaMemcpyToSymbol(allGatherDeviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
  CUDATHROW(cudaMemcpyToSymbol(reduceScatterDeviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
  CUDATHROW(cudaMemcpyToSymbol(ibDeviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
  CUDATHROW(cudaMemcpyToSymbol(globalFlag, &initFlag, sizeof(uint64_t)));
}

std::vector<KernelRestriction> AllReduceTestColl::getKernelRestrictions() {
  return {// {kernelNum, kernelName, compatibleWithMultiNodes, countDivisorForMultiNodes, alignedBytes}
          {0, "allreduce0", true, 1, 4 * worldSize_},
          {1, "allreduce1", true, 1, 4 * worldSize_},
          {2, "allreduce2", true, 1, 2 * worldSize_},
          {3, "allreduce3", true, 3, 4 * worldSize_},
          {
              4,
              "allreduce4",
              true,
              3,
              16 * worldSize_ /*use ulong2 to transfer data*/,
          },
          {5, "allreduce5", false, 1, 4 * worldSize_},
          {6, "allreduce6", false, 1, 4 * worldSize_}
          // {7, "allreduce7", false, 1, 4 * worldSize_}
        };
}

class AllReduceTestEngine : public BaseTestEngine {
 public:
  AllReduceTestEngine(const TestArgs& args);
  ~AllReduceTestEngine() = default;

  void allocateBuffer() override;
  void setupConnections() override;

  bool isUsePacket() const;
  bool isInPlace() const;

  std::vector<void*> getSendBuff() override;
  void* getRecvBuff() override;
  void* getScratchBuff() override;

 private:
  void* getExpectedBuff() override;

  std::shared_ptr<double> inputBuff_;
  std::shared_ptr<double> scratchBuff_;
  std::shared_ptr<double> resultBuff_;
  std::shared_ptr<mscclpp::LLPacket> scratchPacketBuff_;
  std::shared_ptr<mscclpp::LLPacket> putPacketBuff_;
  std::shared_ptr<mscclpp::LLPacket> getPacketBuff_;
  std::shared_ptr<double[]> expectedBuff_;
  std::vector<mscclpp::MemoryChannel> memoryOutOfPlaceChannels_;
  std::vector<mscclpp::MemoryChannel> memoryInPlaceChannels_;
  std::vector<mscclpp::MemoryChannel> memoryOutOfPlaceGetChannels_;
};

AllReduceTestEngine::AllReduceTestEngine(const TestArgs& args) : BaseTestEngine(args, "allreduce") {
  inPlace_ = isInPlace();
}

bool AllReduceTestEngine::isUsePacket() const {
  return (args_.kernelNum == 2 || args_.kernelNum == 6 || args_.kernelNum == 7);
}

bool AllReduceTestEngine::isInPlace() const {
  return (args_.kernelNum != 2 && args_.kernelNum != 6 && args_.kernelNum != 7);
}

void AllReduceTestEngine::allocateBuffer() {
  inputBuff_ = mscclpp::GpuBuffer<double>(args_.maxBytes / sizeof(double)).memory();
  resultBuff_ = mscclpp::GpuBuffer<double>(args_.maxBytes / sizeof(double)).memory();
  inputBuff = inputBuff_.get();
  resultBuff = resultBuff_.get();

  if (args_.kernelNum == 0 || args_.kernelNum == 1 || args_.kernelNum == 3 || args_.kernelNum == 4) {
    scratchBuff_ = mscclpp::GpuBuffer<double>(args_.maxBytes / sizeof(double)).memory();
    scratchBuff = scratchBuff_.get();
  } else if (args_.kernelNum == 2) {
    const size_t nPacket = (args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    // 2x for double-buffering
    const size_t scratchBuffNelem = nPacket * std::max(args_.nRanksPerNode - 1, 1) * 2;
    scratchPacketBuff_ = mscclpp::GpuBuffer<mscclpp::LLPacket>(scratchBuffNelem).memory();
    scratchPacketBuff = scratchPacketBuff_.get();
    const size_t packetBuffNelem = nPacket * 2;
    putPacketBuff_ = mscclpp::GpuBuffer<mscclpp::LLPacket>(packetBuffNelem).memory();
    getPacketBuff_ = mscclpp::GpuBuffer<mscclpp::LLPacket>(packetBuffNelem).memory();
    putPacketBuff = putPacketBuff_.get();
    getPacketBuff = getPacketBuff_.get();
  } else if (args_.kernelNum == 6 || args_.kernelNum == 7) {
    const size_t nPacket = (args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    // 2x for double-buffering, scratchBuff used to store original data and reduced results
    const size_t scratchBuffNelem = nPacket * 2 /*original data & reduced result */ * 2 /* double buffering*/;
    scratchPacketBuff_ = mscclpp::GpuBuffer<mscclpp::LLPacket>(scratchBuffNelem).memory();
    scratchPacketBuff = scratchPacketBuff_.get();
  }

  expectedBuff_ = std::shared_ptr<double[]>(new double[args_.maxBytes / sizeof(double)]);
}

void AllReduceTestEngine::setupConnections() {
  auto getChannelDeviceHandle = [](const std::vector<mscclpp::MemoryChannel>& in,
                                   std::vector<DeviceHandle<mscclpp::MemoryChannel>>& out) {
    return std::transform(in.begin(), in.end(), out.begin(), [](const mscclpp::MemoryChannel& memoryChannel) {
      return mscclpp::deviceHandle(memoryChannel);
    });
  };
  if (isUsePacket()) {
    std::vector<DeviceHandle<mscclpp::PortChannel>> portChannels;

    const size_t nPacket = (args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    if (args_.kernelNum == 6 || args_.kernelNum == 7) {
      const size_t scratchPacketBuffBytes = nPacket * 2 * 2 * sizeof(mscclpp::LLPacket);
      setupMeshConnections(memoryOutOfPlaceChannels_, inputBuff_.get(), args_.maxBytes, scratchPacketBuff_.get(),
                           scratchPacketBuffBytes);
      std::vector<DeviceHandle<mscclpp::MemoryChannel>> memoryChannelDeviceHandles(memoryOutOfPlaceChannels_.size());
      getChannelDeviceHandle(memoryOutOfPlaceChannels_, memoryChannelDeviceHandles);
      CUDATHROW(cudaMemcpyToSymbol(constMemOutOfPlaceChans, memoryChannelDeviceHandles.data(),
                                   sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryChannelDeviceHandles.size()));
    }
    if (args_.kernelNum == 2) {
      const size_t scratchPacketBuffBytes =
          nPacket * std::max(args_.nRanksPerNode - 1, 1) * 2 * sizeof(mscclpp::LLPacket);
      const size_t packetBuffBytes = nPacket * 2 * sizeof(mscclpp::LLPacket);
      setupMeshConnections(memoryOutOfPlaceChannels_, portChannels, inputBuff_.get(), args_.maxBytes,
                           putPacketBuff_.get(), packetBuffBytes, getPacketBuff_.get(), packetBuffBytes,
                           scratchPacketBuff_.get(), scratchPacketBuffBytes);

      if (memoryOutOfPlaceChannels_.size() >
          sizeof(constMemOutOfPlaceChans) / sizeof(DeviceHandle<mscclpp::MemoryChannel>)) {
        std::runtime_error("unexpected error");
      }
      if (portChannels.size() > sizeof(constDevFstRoundChans) / sizeof(DeviceHandle<mscclpp::PortChannel>)) {
        std::runtime_error("unexpected error");
      }

      std::vector<DeviceHandle<mscclpp::MemoryChannel>> memoryChannelDeviceHandles(memoryOutOfPlaceChannels_.size());
      getChannelDeviceHandle(memoryOutOfPlaceChannels_, memoryChannelDeviceHandles);
      CUDATHROW(cudaMemcpyToSymbol(constMemOutOfPlaceChans, memoryChannelDeviceHandles.data(),
                                   sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryChannelDeviceHandles.size()));
      CUDATHROW(cudaMemcpyToSymbol(constDevFstRoundChans, portChannels.data(),
                                   sizeof(DeviceHandle<mscclpp::PortChannel>) * portChannels.size()));
    }
  } else {
    std::vector<DeviceHandle<mscclpp::PortChannel>> fstRoundChannels;
    std::vector<DeviceHandle<mscclpp::PortChannel>> sndRoundChannels;

    // Send data from local inputBuff to remote scratchBuff (out-of-place)
    setupMeshConnections(fstRoundChannels, inputBuff_.get(), args_.maxBytes, scratchBuff_.get(), args_.maxBytes);
    if (fstRoundChannels.size() > sizeof(constDevFstRoundChans) / sizeof(DeviceHandle<mscclpp::PortChannel>)) {
      std::runtime_error("unexpected error");
    }
    CUDATHROW(cudaMemcpyToSymbol(constDevFstRoundChans, fstRoundChannels.data(),
                                 sizeof(DeviceHandle<mscclpp::PortChannel>) * fstRoundChannels.size()));

    // Send data from local inputBuff to remote inputBuff (in-place)
    setupMeshConnections(sndRoundChannels, inputBuff_.get(), args_.maxBytes);
    if (sndRoundChannels.size() > sizeof(constDevSndRoundChans) / sizeof(DeviceHandle<mscclpp::PortChannel>)) {
      std::runtime_error("unexpected error");
    }
    CUDATHROW(cudaMemcpyToSymbol(constDevSndRoundChans, sndRoundChannels.data(),
                                 sizeof(DeviceHandle<mscclpp::PortChannel>) * sndRoundChannels.size()));

    setupMeshConnections(memoryOutOfPlaceChannels_, inputBuff_.get(), args_.maxBytes, scratchBuff_.get(),
                         args_.maxBytes);
    if (memoryOutOfPlaceChannels_.size() >
        sizeof(constMemOutOfPlaceChans) / sizeof(DeviceHandle<mscclpp::MemoryChannel>)) {
      std::runtime_error("unexpected error");
    }
    std::vector<DeviceHandle<mscclpp::MemoryChannel>> memoryChannelDeviceHandles(memoryOutOfPlaceChannels_.size());
    getChannelDeviceHandle(memoryOutOfPlaceChannels_, memoryChannelDeviceHandles);
    CUDATHROW(cudaMemcpyToSymbol(constMemOutOfPlaceChans, memoryChannelDeviceHandles.data(),
                                 sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryChannelDeviceHandles.size()));

    setupMeshConnections(memoryInPlaceChannels_, inputBuff_.get(), args_.maxBytes);
    if (memoryInPlaceChannels_.size() > sizeof(constMemInPlaceChans) / sizeof(DeviceHandle<mscclpp::MemoryChannel>)) {
      std::runtime_error("unexpected error");
    }
    memoryChannelDeviceHandles.resize(memoryInPlaceChannels_.size());
    getChannelDeviceHandle(memoryInPlaceChannels_, memoryChannelDeviceHandles);
    CUDATHROW(cudaMemcpyToSymbol(constMemInPlaceChans, memoryChannelDeviceHandles.data(),
                                 sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryChannelDeviceHandles.size()));

    setupMeshConnections(memoryOutOfPlaceGetChannels_, inputBuff_.get(), args_.maxBytes, scratchBuff_.get(),
                         args_.maxBytes, ChannelSemantic::GET);
    if (memoryOutOfPlaceGetChannels_.size() >
        sizeof(constMemOutOfPlaceGetChans) / sizeof(DeviceHandle<mscclpp::MemoryChannel>)) {
      std::runtime_error("unexpected error");
    }
    memoryChannelDeviceHandles.resize(memoryOutOfPlaceGetChannels_.size());
    getChannelDeviceHandle(memoryOutOfPlaceGetChannels_, memoryChannelDeviceHandles);
    CUDATHROW(cudaMemcpyToSymbol(constMemOutOfPlaceGetChans, memoryChannelDeviceHandles.data(),
                                 sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryChannelDeviceHandles.size()));
  }
}

std::vector<void*> AllReduceTestEngine::getSendBuff() { return {inputBuff_.get()}; }

void* AllReduceTestEngine::getExpectedBuff() { return expectedBuff_.get(); }

void* AllReduceTestEngine::getRecvBuff() { return isInPlace() ? inputBuff_.get() : resultBuff_.get(); }

void* AllReduceTestEngine::getScratchBuff() { return scratchBuff_.get(); }

std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args) {
  return std::make_shared<AllReduceTestEngine>(args);
}

std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<AllReduceTestColl>(); }