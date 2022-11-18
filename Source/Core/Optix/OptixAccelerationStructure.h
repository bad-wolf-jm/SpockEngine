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
        OptixTraversableHandle RTObject = 0;

        OptixTraversableObject() = default;
        OptixTraversableObject( Ref<OptixDeviceContextObject> aRTContext );

        ~OptixTraversableObject() { m_ASBuffer.Dispose(); };

        void AddGeometry( GPUExternalMemory &aVertices, GPUExternalMemory &aIndices, uint32_t aVertexOffset, uint32_t aVertexCount, uint32_t aIndexOffset, uint32_t aIndexCount );

        void Build();

        void Reset();

        GPUMemory &GetBuffer() { return m_ASBuffer; };

      private:
        Ref<OptixDeviceContextObject> m_RTContext = nullptr;

        std::vector<OptixBuildInput> m_TriangleInput = {};
        std::vector<uint32_t> m_InputFlags           = {};

        std::vector<CUdeviceptr> m_VertexBuffers = {};
        std::vector<int32_t> m_VertexCounts      = {};
        std::vector<CUdeviceptr> m_IndexBuffers  = {};
        std::vector<int32_t> m_IndexCounts       = {};

        GPUMemory m_ASBuffer;
    };

} // namespace SE::Graphics
