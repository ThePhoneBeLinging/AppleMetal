#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

int main() {
    using namespace MTL;

    // 1. Get the system default device
    auto device = CreateSystemDefaultDevice();

    // 2. Create command queue
    auto commandQueue = device->newCommandQueue();

    // 3. Load shader from default library (assumes precompiled .metallib)
    auto defaultLibrary = device->newDefaultLibrary();
    auto function = defaultLibrary->newFunction(NS::String::string("simple_shader", NS::StringEncoding::UTF8StringEncoding));

    // 4. Create compute pipeline
    NS::Error* error = nullptr;
    auto pipelineState = device->newComputePipelineState(function, &error);
    if (!pipelineState) {
        std::cerr << "Failed to create pipeline: " << error->localizedDescription()->utf8String() << "\n";
        return -1;
    }

    // 5. Create output buffer
    constexpr int numElements = 16;
    auto bufferSize = numElements * sizeof(float);
    auto outputBuffer = device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);

    // 6. Create command buffer and encoder
    auto commandBuffer = commandQueue->commandBuffer();
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(outputBuffer, 0, 0);

    // 7. Dispatch compute work
    MTL::Size gridSize = MTL::Size(numElements, 1, 1);
    int maxThreads = pipelineState->maxTotalThreadsPerThreadgroup();
    MTL::Size threadgroupSize = MTL::Size(std::min(numElements, maxThreads), 1, 1);
    encoder->dispatchThreads(gridSize, threadgroupSize);

    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // 8. Read back results
    auto data = static_cast<float*>(outputBuffer->contents());
    for (int i = 0; i < numElements; ++i) {
        std::cout << "output[" << i << "] = " << data[i] << "\n";
    }

    // 9. Cleanup (handled by ARC or manually release if using raw pointers)
    return 0;
}
