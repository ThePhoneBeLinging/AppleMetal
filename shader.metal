#include <metal_stdlib>
using namespace metal;

kernel void vectorKernel(device const float3* inVectors  [[ buffer(0) ]],
                         device float4* outVectors       [[ buffer(1) ]],
                         uint id                         [[ thread_position_in_grid ]]) {
    float3 input = inVectors[id];
    float4 output = float4(input.x + 1,input.y,input.z,1);
    outVectors[id] = output;
}
