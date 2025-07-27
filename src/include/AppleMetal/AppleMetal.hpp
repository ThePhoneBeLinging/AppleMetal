//
// Created by Elias Aggergaard Larsen on 27/07-2025.
//

#ifndef APPLEMETAL_H
#define APPLEMETAL_H
#include "Metal/MTLCommandQueue.hpp"
#include "Metal/MTLComputePipeline.hpp"


class AppleMetal
{
public:
  AppleMetal();
  ~AppleMetal() = default;

  std::vector<double> computeWithShader(const std::vector<double>& radius, const std::vector<double>& points);

private:
  MTL::Device* device_;
  MTL::CommandQueue* commandQueue_;
  MTL::ComputePipelineState* pipelineState_;
};


#endif //APPLEMETAL_H
