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

__device__ mscclpp::DeviceSyncer deviceSyncer;
__device__ mscclpp::DeviceSyncer allGatherDeviceSyncer;
__device__ mscclpp::DeviceSyncer reduceScatterDeviceSyncer;
__device__ mscclpp::DeviceSyncer ibDeviceSyncer;

__global__ void __launch_bounds__(1024)
    allreduce2(double* buff, void* scratch, void* putPktBuf, void* getPktBuf, void* result, int rank, int nRanksPerNode,
               int worldSize, size_t nelems) {
  int numPeersPerNode = nRanksPerNode - 1;
  size_t nPkts = nelems / 2;  // 2 elems per packet, assume nelems is even
  size_t pktBytes = nPkts * sizeof(mscclpp::LLPacket);

  // Channel to a local peer
  int memChanIdx = blockIdx.x / BLOCKS_PER_PEER;
  DeviceHandle<mscclpp::MemoryChannel> memChan = constMemOutOfPlaceChans[memChanIdx];

  // Channel to a remote peer that has the same local rank as me
  int localRank = rank % nRanksPerNode;
  DeviceHandle<mscclpp::PortChannel> portChan = constDevFstRoundChans[localRank];

  // Flag for packets. Initially 1
  uint32_t flag = (uint32_t)globalFlag;

  double2* src = (double2*)buff;
  double2* res = (double2*)result;
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
      // Cast doubles to uint64_t for packet writing
      uint64_t x_bits = __double_as_longlong(src[idx].x);
      uint64_t y_bits = __double_as_longlong(src[idx].y);
      putPktPtr[idx].write((uint32_t)(x_bits & 0xFFFFFFFF), (uint32_t)(x_bits >> 32), flag);
      putPktPtr[idx + nPkts].write((uint32_t)(y_bits & 0xFFFFFFFF), (uint32_t)(y_bits >> 32), flag);
    }
  } else {
    // Offset of the input data (buff) to read from
    size_t srcOffset =
        ((blockIdx.x % BLOCKS_PER_PEER) * nelems * sizeof(double) / BLOCKS_PER_PEER);  // offset for this block
    // Offset of the peer's scratch buffer (scratch) to write on
    size_t dstOffset = (scratchOffset) +                                                    // double buffering
                       ((memChanIdx < localRank ? localRank - 1 : localRank) * pktBytes) +  // offset for this rank
                       (srcOffset * 2);  // offset for this block: twice of srcOffset because 2 elems per packet
    // Write data to the peer's scratch
    memChan.putPackets(dstOffset, srcOffset, nelems / BLOCKS_PER_PEER * sizeof(double), threadIdx.x, blockDim.x, flag);
    // Read data from my scratch, reduce data with my buff, and write the result to my putPktBuf or to result
    const bool isSingleNode = (worldSize == nRanksPerNode);
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * gridDim.x) {
      double x = 0.0;
      double y = 0.0;
      for (int peerIdx = 0; peerIdx < numPeersPerNode / 2; ++peerIdx) {
        uint2 data0 = memChan.unpackPacket(scratchBaseIndex + 2 * peerIdx * nPkts + idx, flag);
        uint2 data1 = memChan.unpackPacket(scratchBaseIndex + (2 * peerIdx + 1) * nPkts + idx, flag);
        // Reconstruct doubles from two uint32_t values
        uint64_t bits0 = ((uint64_t)data0.y << 32) | data0.x;
        uint64_t bits1 = ((uint64_t)data1.y << 32) | data1.x;
        x += __longlong_as_double(bits0);
        y += __longlong_as_double(bits1);
      }
      if (numPeersPerNode & 1) {
        uint2 data = memChan.unpackPacket(scratchBaseIndex + (numPeersPerNode - 1) * nPkts + idx, flag);
        uint64_t bits = ((uint64_t)data.y << 32) | data.x;
        x += __longlong_as_double(bits);
      }
      if (isSingleNode) {
        res[idx].x = src[idx].x + x;
        res[idx].y = src[idx].y + y;
      } else {
        // Cast result doubles to uint64_t for packet writing
        uint64_t result_x_bits = __double_as_longlong(src[idx].x + x);
        uint64_t result_y_bits = __double_as_longlong(src[idx].y + y);
        putPktPtr[idx].write((uint32_t)(result_x_bits & 0xFFFFFFFF), (uint32_t)(result_x_bits >> 32), flag);
        putPktPtr[idx + nPkts].write((uint32_t)(result_y_bits & 0xFFFFFFFF), (uint32_t)(result_y_bits >> 32), flag);
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
      portChan.put(pktBufOffset, pktBytes * 2);  // *2 because we now store each double in 2 packets
      if ((flag & 63) == 0) {
        portChan.flush();
      }
    }

    // Read data from my getPktBuf, reduce data with my putPktBuf, and write the result to result
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * nBlocksPhase2) {
      // Read x values
      uint2 data0_x = putPktPtr[idx].read(flag);
      uint2 data1_x = getPktPtr[idx].read(flag);
      uint64_t bits0_x = ((uint64_t)data0_x.y << 32) | data0_x.x;
      uint64_t bits1_x = ((uint64_t)data1_x.y << 32) | data1_x.x;
      
      // Read y values
      uint2 data0_y = putPktPtr[idx + nPkts].read(flag);
      uint2 data1_y = getPktPtr[idx + nPkts].read(flag);
      uint64_t bits0_y = ((uint64_t)data0_y.y << 32) | data0_y.x;
      uint64_t bits1_y = ((uint64_t)data1_y.y << 32) | data1_y.x;
      
      res[idx].x = __longlong_as_double(bits0_x) + __longlong_as_double(bits1_x);
      res[idx].y = __longlong_as_double(bits0_y) + __longlong_as_double(bits1_y);
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

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

  int nBlocks = std::max(args.nRanksPerNode - 1, 1) * BLOCKS_PER_PEER;
  void* tmpBuff = scratchPacketBuff;
  int nThreadsPerBlock = 1024;

  if (kernelNum == 2)
    allreduce2<<<nBlocks, nThreadsPerBlock, 0, stream>>>((double*)inputBuff, tmpBuff, putPacketBuff, getPacketBuff,
                                                         resultBuff, rank, args.nRanksPerNode, worldSize, paramCount_);
}

void AllReduceTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  if (sendBuff.size() != 1) std::runtime_error("unexpected error");
  const int rank = args.rank;
  const int worldSize = args.totalRanks;
  std::vector<double> dataHost(std::max(sendCount_, recvCount_), (double)rank);
  CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), sendCount_ * typeSize_, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = (double)(worldSize * (worldSize - 1)) / 2.0;
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
  return {
          {2, "allreduce2", true, 1, 2 * worldSize_}  // 8 bytes for double instead of 4 for int
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
  return (args_.kernelNum == 2);
}

bool AllReduceTestEngine::isInPlace() const {
  return (args_.kernelNum != 2);
}

void AllReduceTestEngine::allocateBuffer() {
  inputBuff_ = mscclpp::GpuBuffer<double>(args_.maxBytes / sizeof(double)).memory();
  resultBuff_ = mscclpp::GpuBuffer<double>(args_.maxBytes / sizeof(double)).memory();
  inputBuff = inputBuff_.get();
  resultBuff = resultBuff_.get();

  const size_t nPacket = (args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  // 2x for double-buffering, 2x because each double needs 2 packets (64 bits = 2x32 bits)
  const size_t scratchBuffNelem = nPacket * std::max(args_.nRanksPerNode - 1, 1) * 2;
  scratchPacketBuff_ = mscclpp::GpuBuffer<mscclpp::LLPacket>(scratchBuffNelem).memory();
  scratchPacketBuff = scratchPacketBuff_.get();
  // 2x for double-buffering, 2x because each double needs 2 packets
  const size_t packetBuffNelem = nPacket * 2 * 2;
  putPacketBuff_ = mscclpp::GpuBuffer<mscclpp::LLPacket>(packetBuffNelem).memory();
  getPacketBuff_ = mscclpp::GpuBuffer<mscclpp::LLPacket>(packetBuffNelem).memory();
  putPacketBuff = putPacketBuff_.get();
  getPacketBuff = getPacketBuff_.get();

  expectedBuff_ = std::shared_ptr<double[]>(new double[args_.maxBytes / sizeof(double)]);
}

void AllReduceTestEngine::setupConnections() {
  auto getChannelDeviceHandle = [](const std::vector<mscclpp::MemoryChannel>& in,
                                   std::vector<DeviceHandle<mscclpp::MemoryChannel>>& out) {
    return std::transform(in.begin(), in.end(), out.begin(), [](const mscclpp::MemoryChannel& memoryChannel) {
      return mscclpp::deviceHandle(memoryChannel);
    });
  };

  std::vector<DeviceHandle<mscclpp::PortChannel>> portChannels;

  const size_t nPacket = (args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  const size_t scratchPacketBuffBytes =
      nPacket * std::max(args_.nRanksPerNode - 1, 1) * 2 * sizeof(mscclpp::LLPacket);
  const size_t packetBuffBytes = nPacket * 2 * 2 * sizeof(mscclpp::LLPacket);  // 2x for double storage
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

std::vector<void*> AllReduceTestEngine::getSendBuff() { return {inputBuff_.get()}; }

void* AllReduceTestEngine::getExpectedBuff() { return expectedBuff_.get(); }

void* AllReduceTestEngine::getRecvBuff() { return isInPlace() ? inputBuff_.get() : resultBuff_.get(); }

void* AllReduceTestEngine::getScratchBuff() { return scratchBuff_.get(); }

std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args) {
  return std::make_shared<AllReduceTestEngine>(args);
}

std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<AllReduceTestColl>(); }