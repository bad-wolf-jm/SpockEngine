#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Core/CUDA/Array/CudaBuffer.h"
#include "Scene/VertexData.h"

#include "Optix7.h"
#include "OptixContext.h"
#include "OptixProgramGroup.h"
#include "OptixShaderBindingTable.h"

#include "Graphics/VkGpuBuffer.h"

namespace SE::Graphics
{
    using namespace SE::Cuda;
    using namespace SE::Core;

    struct OptixScene
    {
        OptixTraversableHandle mOptixObject = 0;

        OptixScene() = default;
        OptixScene( Ref<OptixDeviceContextObject> aRayTracingContext );

        ~OptixScene() { mAccelerationStructureBuffer.Dispose(); };

        template <typename _VertexStructType>
        void AddGeometry( GPUMemory &aVertices, GPUMemory &aIndices, uint32_t aVertexOffset, uint32_t aVertexCount,
                          uint32_t aIndexOffset, uint32_t aIndexCount )
        {
            mVertexBuffers.push_back( (CUdeviceptr)( aVertices.DataAs<_VertexStructType>() + aVertexOffset ) );
            mVertexCounts.push_back( (int)aVertexCount );
            mVertexStrides.push_back( sizeof( _VertexStructType ) );
            mIndexBuffers.push_back( (CUdeviceptr)( aIndices.DataAs<uint32_t>() + aIndexOffset ) );
            mIndexCounts.push_back( (int)( aIndexCount / 3 ) );
        }

        void AddGeometry( VkGpuBuffer &aVertices, VkGpuBuffer &aIndices, uint32_t aVertexOffset, uint32_t aVertexCount,
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
        std::vector<int32_t>     mVertexStrides = {};
        std::vector<CUdeviceptr> mIndexBuffers  = {};
        std::vector<int32_t>     mIndexCounts   = {};

        GPUMemory mAccelerationStructureBuffer;
    };

} // namespace SE::Graphics
