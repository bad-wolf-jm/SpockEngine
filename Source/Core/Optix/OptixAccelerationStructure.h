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

#include "Graphics/Vulkan/VkGpuBuffer.h"

namespace SE::Graphics
{
    using namespace SE::Cuda;
    using namespace SE::Core;

    struct OptixScene
    {
        OptixTraversableHandle mOptixObject = 0;

        OptixScene() = default;
        OptixScene( ref_t<OptixDeviceContextObject> aRayTracingContext );

        ~OptixScene() { mAccelerationStructureBuffer.Dispose(); };

        template <typename _VertexStructType>
        void AddGeometry( GPUMemory &aVertices, GPUMemory &aIndices, uint32_t aVertexOffset, uint32_t aVertexCount,
                          uint32_t aIndexOffset, uint32_t aIndexCount )
        {
            mVertexBuffers.push_back( (RawPointer)( aVertices.DataAs<_VertexStructType>() + aVertexOffset ) );
            mVertexCounts.push_back( (int)aVertexCount );
            mVertexStrides.push_back( sizeof( _VertexStructType ) );
            mIndexBuffers.push_back( (RawPointer)( aIndices.DataAs<uint32_t>() + aIndexOffset ) );
            mIndexCounts.push_back( (int)( aIndexCount / 3 ) );
        }

        void AddGeometry( VkGpuBuffer &aVertices, VkGpuBuffer &aIndices, uint32_t aVertexOffset, uint32_t aVertexCount,
                          uint32_t aIndexOffset, uint32_t aIndexCount );

        void Build();

        void Reset();

        GPUMemory &GetBuffer() { return mAccelerationStructureBuffer; };

      private:
        ref_t<OptixDeviceContextObject> mRayTracingContext = nullptr;

        vector_t<OptixBuildInput> mTriangleInput = {};
        vector_t<uint32_t>        mInputFlags    = {};

        vector_t<RawPointer> mVertexBuffers = {};
        vector_t<int32_t>    mVertexCounts  = {};
        vector_t<int32_t>    mVertexStrides = {};
        vector_t<RawPointer> mIndexBuffers  = {};
        vector_t<int32_t>    mIndexCounts   = {};

        GPUMemory mAccelerationStructureBuffer;
    };

} // namespace SE::Graphics
