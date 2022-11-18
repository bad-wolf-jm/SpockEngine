#pragma once

#include "Core/Math/Types.h"

// #include "Core/Cuda/CudaArray.h"
#include "Core/Cuda/CudaBuffer.h"
// #include "SensorModelDev/Base/KernelComponents.h"

#include "Scene/VertexData.h"

using namespace SE::Cuda;
using namespace SE::Core;

namespace SE::Graphics
{

    void StaticVertexTransform( VertexData *aOutTransformedVertices, VertexData *aVertices,
        math::mat4 *aObjectToWorldTransform, uint32_t aObjectCount, uint32_t *aObjectOffsets, uint32_t *aObjectVertexCount,
        uint32_t aMaxVertexCount );
        
    void SkinnedVertexTransform( VertexData *aOutTransformedVertices, VertexData *aVertices,
        math::mat4 *aObjectToWorldTransform, math::mat4 *aJointMatrices, uint32_t *aJointOffsets, uint32_t aObjectCount,
        uint32_t *aObjectOffsets, uint32_t *aObjectVertexCount, uint32_t aMaxVertexCount );

} // namespace SE::Graphics