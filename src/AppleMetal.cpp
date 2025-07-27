//
// Created by Elias Aggergaard Larsen on 27/07-2025.
//

#include "AppleMetal/AppleMetal.hpp"

#include <iostream>
#include <simd/vector_types.h>

#include "Metal/MTLCommandBuffer.hpp"
#include "Metal/MTLComputeCommandEncoder.hpp"
#include "Metal/MTLDevice.hpp"
#include "Metal/MTLLibrary.hpp"
#include "Metal/MTLBuffer.hpp"

AppleMetal::AppleMetal()
{
  device_ = MTL::CreateSystemDefaultDevice();

  commandQueue_ = device_->newCommandQueue();

  auto defaultLibrary = device_->newDefaultLibrary();
  auto function = defaultLibrary->newFunction(
    NS::String::string("vectorKernel", NS::StringEncoding::UTF8StringEncoding));

  NS::Error* error = nullptr;
  pipelineState_ = device_->newComputePipelineState(function, &error);
  if (!pipelineState_)
  {
    std::cerr << "Failed to create pipeline: " << error->localizedDescription()->utf8String() << "\n";
  }
}

std::vector<double> AppleMetal::computeWithShader(const std::vector<double>& radius, const std::vector<double>& points)
{
  constexpr int numElements = 1280 * 720;
  auto inputBufferSize = numElements * sizeof(simd::float3);
  auto inputBuffer = device_->newBuffer(inputBufferSize, MTL::ResourceStorageModeShared);
  auto input = static_cast<simd::float3*>(inputBuffer->contents());
  for (int i = 0; i < numElements; i++)
  {
    input[i] = simd::float3{1, 1, 1};
  }

  auto outputBufferSize = numElements * sizeof(simd::float4);
  auto outputBuffer = device_->newBuffer(outputBufferSize, MTL::ResourceStorageModeShared);

  auto commandBuffer = commandQueue_->commandBuffer();
  auto encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipelineState_);
  encoder->setBuffer(inputBuffer, 0, 0);
  encoder->setBuffer(outputBuffer, 0, 1);

  MTL::Size gridSize = MTL::Size(numElements, 1, 1);
  int maxThreads = pipelineState_->maxTotalThreadsPerThreadgroup();
  MTL::Size threadgroupSize = MTL::Size(std::min(numElements, maxThreads), 1, 1);
  encoder->dispatchThreads(gridSize, threadgroupSize);

  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  // 8. Read back results
  auto data = static_cast<simd::float4*>(outputBuffer->contents());
  std::vector<double> result(numElements * 4);
  for (int i = 0; i < numElements * 4; i += 4)
  {
    result[i] = data[i].x;
    result[i + 1] = data[i].y;
    result[i + 2] = data[i].z;
    result[i + 3] = data[i].w;
  }

  return result;
}
