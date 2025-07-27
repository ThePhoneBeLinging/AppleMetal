//
// Created by Elias Aggergaard Larsen on 27/07-2025.
//

#ifndef APPLEMETAL_H
#define APPLEMETAL_H
#include "LibDataTypes/Ray.h"
#include "LibDataTypes/Sphere.h"
#include "Metal/MTLCommandQueue.hpp"
#include "Metal/MTLComputePipeline.hpp"


class AppleMetal
{
public:
  AppleMetal();
  ~AppleMetal() = default;

  std::vector<EAL::Double3> computeWithShader(const std::vector<EAL::Ray>& rays,
                                              const std::vector<EAL::Sphere>& spheres);

private:
  MTL::Device* device_;
  MTL::CommandQueue* commandQueue_;
  MTL::ComputePipelineState* pipelineState_;
};


#endif //APPLEMETAL_H
