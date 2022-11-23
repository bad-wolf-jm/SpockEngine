#include "OptixAccelerationStructure.h"

namespace SE::Graphics
{

    OptixScene::OptixScene( Ref<OptixDeviceContextObject> a_RTContext )
        : mRayTracingContext{ a_RTContext }
    {
    }

    void OptixScene::AddGeometry( GPUExternalMemory &aVertices, GPUExternalMemory &aIndices, uint32_t aVertexOffset,
                                  uint32_t aVertexCount, uint32_t aIndexOffset, uint32_t aIndexCount )
    {
        mVertexBuffers.push_back( (CUdeviceptr)( aVertices.DataAs<VertexData>() + aVertexOffset ) );
        mVertexCounts.push_back( (int)aVertexCount );
        mVertexStrides.push_back( sizeof(VertexData) );
        mIndexBuffers.push_back( (CUdeviceptr)( aIndices.DataAs<uint32_t>() + aIndexOffset ) );
        mIndexCounts.push_back( (int)( aIndexCount / 3 ) );
    }

    void OptixScene::Build()
    {
        for( uint32_t i = 0; i < mVertexBuffers.size(); i++ )
        {
            mTriangleInput.emplace_back( OptixBuildInput{} );
            mInputFlags.push_back( 0 );
            auto &lBuildinput = mTriangleInput.back(); //[mTriangleInput.size() - 1];

            lBuildinput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            lBuildinput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
            lBuildinput.triangleArray.vertexStrideInBytes = mVertexStrides[i];
            lBuildinput.triangleArray.numVertices         = mVertexCounts[i];
            lBuildinput.triangleArray.vertexBuffers       = &( mVertexBuffers.data()[i] );
            lBuildinput.triangleArray.preTransform        = 0;

            lBuildinput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            lBuildinput.triangleArray.indexStrideInBytes = sizeof( math::uvec3 );
            lBuildinput.triangleArray.numIndexTriplets   = mIndexCounts[i];
            lBuildinput.triangleArray.indexBuffer        = mIndexBuffers[i];

            lBuildinput.triangleArray.flags                       = 0;
            lBuildinput.triangleArray.numSbtRecords               = 1;
            lBuildinput.triangleArray.sbtIndexOffsetBuffer        = 0;
            lBuildinput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
            lBuildinput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
        }

        uint32_t l_Idx = 0;
        for( auto &lBuildinput : mTriangleInput ) lBuildinput.triangleArray.flags = &mInputFlags.data()[l_Idx++];

        OptixAccelBuildOptions lAccelOptions{};
        lAccelOptions.buildFlags            = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        lAccelOptions.motionOptions.numKeys = 1;
        lAccelOptions.operation             = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes lBlasBufferSizes{};
        OPTIX_CHECK( optixAccelComputeMemoryUsage( mRayTracingContext->mOptixObject, &lAccelOptions, mTriangleInput.data(),
                                                   (int)mTriangleInput.size(), &lBlasBufferSizes ) );

        GPUMemory lCompactedSizeBuffer( sizeof( uint64_t ) );
        GPUMemory lTempBuffer( lBlasBufferSizes.tempSizeInBytes );
        GPUMemory lOutputBuffer( lBlasBufferSizes.outputSizeInBytes );

        OptixAccelEmitDesc lEmitDesc{};
        lEmitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        lEmitDesc.result = lCompactedSizeBuffer.RawDevicePtr();

        OPTIX_CHECK( optixAccelBuild( mRayTracingContext->mOptixObject, 0, &lAccelOptions, mTriangleInput.data(),
                                      (int)mTriangleInput.size(), lTempBuffer.RawDevicePtr(), lTempBuffer.Size(),
                                      lOutputBuffer.RawDevicePtr(), lOutputBuffer.Size(), &mOptixObject, &lEmitDesc, 1 ) );
        CUDA_SYNC_CHECK();

        uint64_t lCompactedSize = lCompactedSizeBuffer.Fetch<uint64_t>()[0];

        mAccelerationStructureBuffer = GPUMemory( lCompactedSize );
        OPTIX_CHECK( optixAccelCompact( mRayTracingContext->mOptixObject, 0, mOptixObject, mAccelerationStructureBuffer.RawDevicePtr(),
                                        mAccelerationStructureBuffer.Size(), &mOptixObject ) );

        CUDA_SYNC_CHECK();

        lTempBuffer.Dispose();
        lOutputBuffer.Dispose();
        lCompactedSizeBuffer.Dispose();
    }

    void OptixScene::Reset() {}

} // namespace SE::Graphics
