#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Core/Cuda/ExternalMemory.h"
#include "Scene/VertexData.h"

#include "Optix7.h"
#include "OptixContext.h"
#include "OptixProgramGroup.h"
#include "OptixShaderBindingTable.h"

namespace SE::Graphics
{
    using namespace SE::Cuda;
    using namespace SE::Core;

    struct OptixTraversableObject
    {
        OptixTraversableHandle mOptixObject = 0;

        OptixTraversableObject() = default;
        OptixTraversableObject( Ref<OptixDeviceContextObject> aRayTracingContext );

        ~OptixTraversableObject() { mAccelerationStructureBuffer.Dispose(); };

        void AddGeometry( GPUExternalMemory &aVertices, GPUExternalMemory &aIndices, uint32_t aVertexOffset, uint32_t aVertexCount,
                          uint32_t aIndexOffset, uint32_t aIndexCount );

        void Build();

        void Reset();

        GPUMemory &GetBuffer() { return mAccelerationStructureBuffer; };

      private:
        Ref<OptixDeviceContextObject> mRayTracingContext = nullptr;

        std::vector<OptixBuildInput> mTriangleInput = {};
        std::vector<uint32_t>        mInputFlags    = {};

        std::vector<CUdeviceptr> mVertexBuffers = {};
        std::vector<int32_t>     mVertexCounts  = {};
        std::vector<CUdeviceptr> mIndexBuffers  = {};
        std::vector<int32_t>     mIndexCounts   = {};

        GPUMemory mAccelerationStructureBuffer;
    };

} // namespace SE::Graphics
