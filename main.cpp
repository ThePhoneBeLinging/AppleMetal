#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "simd/simd.h"

int main()
{
  using namespace MTL;

  // 1. Get the system default device
  auto device = CreateSystemDefaultDevice();

  // 2. Create command queue
  auto commandQueue = device->newCommandQueue();

  // 3. Load shader from default library (assumes precompiled .metallib)
  auto defaultLibrary = device->newDefaultLibrary();
  auto function = defaultLibrary->newFunction(
    NS::String::string("vectorKernel", NS::StringEncoding::UTF8StringEncoding));

  // 4. Create compute pipeline
  NS::Error* error = nullptr;
  auto pipelineState = device->newComputePipelineState(function, &error);
  if (!pipelineState)
  {
    std::cerr << "Failed to create pipeline: " << error->localizedDescription()->utf8String() << "\n";
    return -1;
  }

  // 5. Create output buffer
  constexpr int numElements = 1280 * 720;
  auto inputBufferSize = numElements * sizeof(simd::float3);
  auto inputBuffer = device->newBuffer(inputBufferSize, MTL::ResourceStorageModeShared);
  auto input = static_cast<simd::float3*>(inputBuffer->contents());
  for (int i = 0; i < numElements; i++)
  {
    input[i] = simd::float3{1, 1, 1};
  }

  auto outputBufferSize = numElements * sizeof(simd::float4);
  auto outputBuffer = device->newBuffer(outputBufferSize, MTL::ResourceStorageModeShared);

  // 6. Create command buffer and encoder
  auto commandBuffer = commandQueue->commandBuffer();
  auto encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState);
  encoder->setBuffer(inputBuffer, 0, 0);
  encoder->setBuffer(outputBuffer, 0, 1);

  // 7. Dispatch compute work
  MTL::Size gridSize = MTL::Size(numElements, 1, 1);
  int maxThreads = pipelineState->maxTotalThreadsPerThreadgroup();
  MTL::Size threadgroupSize = MTL::Size(std::min(numElements, maxThreads), 1, 1);
  encoder->dispatchThreads(gridSize, threadgroupSize);

  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  // 8. Read back results
  auto data = static_cast<simd::float4*>(outputBuffer->contents());
  for (int i = 0; i < numElements; ++i)
  {
    auto localData = data[i];
    if (localData.x != 0)
    {
      std::cout << "output[" << i << "] = " << data[i].x << "\n";
    }
  }

  // 9. Cleanup (handled by ARC or manually release if using raw pointers)
  return 0;
}
