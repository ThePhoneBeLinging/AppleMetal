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
#include "Metal/MTLTexture.hpp"

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

void AppleMetal::computeWithShader(const std::vector<EAL::Ray>& rays,
                                   const std::vector<EAL::Sphere>& spheres, EAL::Image* image)
{
  int numElements = rays.size();
  auto inputBufferSize = numElements * sizeof(simd::float3);
  auto inputBuffer = device_->newBuffer(inputBufferSize, MTL::ResourceStorageModeShared);
  auto input = static_cast<simd::float3*>(inputBuffer->contents());
  for (int i = 0; i < numElements; i++)
  {
    input[i].x = rays[i].vector.x;
    input[i].y = rays[i].vector.y;
    input[i].z = rays[i].vector.z;
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
  auto data = static_cast<simd::float3*>(outputBuffer->contents());
  for (int i = 0; i < numElements; i++)
  {
    image->pixelBuffer_[i].r = data[i].x;
    image->pixelBuffer_[i].g = data[i].y;
    image->pixelBuffer_[i].b = data[i].z;
  }
}
