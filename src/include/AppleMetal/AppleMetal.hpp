//
// Created by Elias Aggergaard Larsen on 27/07-2025.
//

#ifndef APPLEMETAL_H
#define APPLEMETAL_H
#include "LibDataTypes/Ray.h"
#include "LibDataTypes/Sphere.h"
#include "Metal/MTLTexture.hpp"
#include "Metal/MTLCommandQueue.hpp"
#include "Metal/MTLComputePipeline.hpp"
#include "LibDataTypes/IComputeShader.h"
#include "LibDataTypes/Image.h"
#include "Metal/MTLBuffer.hpp"


class AppleMetal : public EAL::IComputeShader
{
public:
  AppleMetal();
  ~AppleMetal() override = default;

  void computeWithShader(const std::vector<EAL::Ray>& rays,
                         const std::vector<EAL::Sphere>& spheres, EAL::Image* image) override;

private:
  MTL::Device* device_;
  MTL::CommandQueue* commandQueue_;
  MTL::ComputePipelineState* pipelineState_;
  MTL::Buffer* rayBuffer_;
  MTL::Buffer* outputBuffer_;
};


#endif //APPLEMETAL_H
