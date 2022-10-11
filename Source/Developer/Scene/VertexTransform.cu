#include "Cuda/CudaAssert.h"
#include "VertexTransform.h"

#include "TensorOps/Implementation/HelperMacros.h"
namespace LTSE::Graphics
{

#define THREADS_PER_BLOCK 512

    namespace Kernels
    {
        CUDA_KERNEL_DEFINITION void SkinnedVertexTransform( VertexData *aOutTransformedVertices, VertexData *aVertices, math::mat4 *aObjectToWorldTransform,
                                                            math::mat4 *aJointMatrices, uint32_t aJointCount, uint32_t aObjectCount, uint32_t *aObjectOffsets,
                                                            uint32_t *aObjectVertexCount )
        {
            uint32_t lObjectOffset      = aObjectOffsets[blockIdx.x];
            uint32_t lObjectVertexCount = aObjectVertexCount[blockIdx.x];

            uint32_t lVertexID = blockIdx.y * LTSE::TensorOps::Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( lVertexID < lObjectVertexCount );

            math::mat4 lTransform = aObjectToWorldTransform[blockIdx.x];

            VertexData lVertex = aVertices[lObjectOffset + lVertexID];

            math::mat4 lSkinTransform = lVertex.Weights.x * aJointMatrices[int( lVertex.Bones.x )] + lVertex.Weights.y * aJointMatrices[int( lVertex.Bones.y )] +
                                        lVertex.Weights.z * aJointMatrices[int( lVertex.Bones.z )] + lVertex.Weights.w * aJointMatrices[int( lVertex.Bones.w )];

            math::mat4 lFinalTransform = lTransform * lSkinTransform;

            aOutTransformedVertices[lObjectOffset + lVertexID] = lVertex;

            aOutTransformedVertices[lObjectOffset + lVertexID].Position = lFinalTransform * lVertex.Position;
            aOutTransformedVertices[lObjectOffset + lVertexID].Normal   = normalize( transpose( inverse( mat3( lFinalTransform ) ) ) * lVertex.Normal );
        }

        CUDA_KERNEL_DEFINITION void StaticVertexTransform( VertexData *aOutTransformedVertices, VertexData *aVertices, math::mat4 *aObjectToWorldTransform, uint32_t aObjectCount,
                                                           uint32_t *aObjectOffsets, uint32_t *aObjectVertexCount )
        {
            uint32_t lObjectOffset      = aObjectOffsets[blockIdx.x];
            uint32_t lObjectVertexCount = aObjectVertexCount[blockIdx.x];

            uint32_t lVertexID = blockIdx.y * LTSE::TensorOps::Private::ThreadsPerBlock + threadIdx.x;

            RETURN_UNLESS( lVertexID < lObjectVertexCount );

            math::mat4 lTransform = aObjectToWorldTransform[blockIdx.x];
            VertexData lVertex    = aVertices[lObjectOffset + lVertexID];

            aOutTransformedVertices[lObjectOffset + lVertexID] = lVertex;

            aOutTransformedVertices[lObjectOffset + lVertexID].Position = lTransform * lVertex.Position;
            aOutTransformedVertices[lObjectOffset + lVertexID].Normal   = normalize( transpose( inverse( mat3( lTransform ) ) ) * lVertex.Normal );
        }
    } // namespace Kernels

    void StaticVertexTransform( VertexData *aOutTransformedVertices, VertexData *aVertices, math::mat4 *aObjectToWorldTransform, uint32_t aObjectCount, uint32_t *aObjectOffsets,
                                uint32_t *aObjectVertexCount, uint32_t aMaxVertexCount )
    {
        int lBlockCount = ( aMaxVertexCount / LTSE::TensorOps::Private::ThreadsPerBlock ) + 1;
        dim3 lGridDim( aObjectCount, lBlockCount, 1 );
        dim3 lBlockDim( LTSE::TensorOps::Private::ThreadsPerBlock );

        Kernels::StaticVertexTransform<<<lGridDim, lBlockDim>>>( aOutTransformedVertices, aVertices, aObjectToWorldTransform, aObjectCount, aObjectOffsets, aObjectVertexCount );
    }

    void SkinnedVertexTransform( VertexData *aOutTransformedVertices, VertexData *aVertices, math::mat4 *aObjectToWorldTransform, math::mat4 *aJointMatrices, uint32_t aJointCount,
                                 uint32_t aObjectCount, uint32_t *aObjectOffsets, uint32_t *aObjectVertexCount, uint32_t aMaxVertexCount )
    {
        int lBlockCount = ( aMaxVertexCount / LTSE::TensorOps::Private::ThreadsPerBlock ) + 1;
        dim3 lGridDim( aObjectCount, lBlockCount, 1 );
        dim3 lBlockDim( LTSE::TensorOps::Private::ThreadsPerBlock );

        Kernels::SkinnedVertexTransform<<<lGridDim, lBlockDim>>>( aOutTransformedVertices, aVertices, aObjectToWorldTransform, aJointMatrices, aJointCount, aObjectCount,
                                                                  aObjectOffsets, aObjectVertexCount );
    }

} // namespace LTSE::Graphics