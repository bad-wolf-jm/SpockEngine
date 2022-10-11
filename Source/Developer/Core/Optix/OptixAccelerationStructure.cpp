#include "OptixAccelerationStructure.h"

namespace LTSE::Graphics
{

    OptixTraversableObject::OptixTraversableObject( Ref<OptixDeviceContextObject> a_RTContext )
        : m_RTContext{ a_RTContext }
    {
    }

    void OptixTraversableObject::AddGeometry( GPUExternalMemory &aVertices, GPUExternalMemory &aIndices, uint32_t aVertexOffset, uint32_t aVertexCount, uint32_t aIndexOffset,
                                              uint32_t aIndexCount )
    {
        m_VertexBuffers.push_back( (CUdeviceptr)( aVertices.DataAs<VertexData>() + aVertexOffset ) );
        m_VertexCounts.push_back( (int)aVertexCount );
        m_IndexBuffers.push_back( (CUdeviceptr)( aIndices.DataAs<uint32_t>() + aIndexOffset ) );
        m_IndexCounts.push_back( (int)( aIndexCount / 3 ) );
    }

    void OptixTraversableObject::Build()
    {
        for( uint32_t i = 0; i < m_VertexBuffers.size(); i++ )
        {
            m_TriangleInput.emplace_back( OptixBuildInput{} );
            m_InputFlags.push_back( 0 );
            auto &l_Buildinput = m_TriangleInput[m_TriangleInput.size() - 1];

            l_Buildinput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            l_Buildinput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
            l_Buildinput.triangleArray.vertexStrideInBytes = sizeof( VertexData );
            l_Buildinput.triangleArray.numVertices         = m_VertexCounts[i];
            l_Buildinput.triangleArray.vertexBuffers       = &( m_VertexBuffers.data()[i] );
            l_Buildinput.triangleArray.preTransform        = 0;

            l_Buildinput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            l_Buildinput.triangleArray.indexStrideInBytes = sizeof( math::uvec3 );
            l_Buildinput.triangleArray.numIndexTriplets   = m_IndexCounts[i];
            l_Buildinput.triangleArray.indexBuffer        = m_IndexBuffers[i];

            l_Buildinput.triangleArray.flags                       = 0;
            l_Buildinput.triangleArray.numSbtRecords               = 1;
            l_Buildinput.triangleArray.sbtIndexOffsetBuffer        = 0;
            l_Buildinput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
            l_Buildinput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
        }

        uint32_t l_Idx = 0;
        for( auto &l_Buildinput : m_TriangleInput )
            l_Buildinput.triangleArray.flags = &m_InputFlags.data()[l_Idx++];

        OptixAccelBuildOptions l_AccelOptions = {};
        l_AccelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        l_AccelOptions.motionOptions.numKeys  = 1;
        l_AccelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes l_BlasBufferSizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( m_RTContext->RTObject, &l_AccelOptions, m_TriangleInput.data(), (int)m_TriangleInput.size(), &l_BlasBufferSizes ) );

        GPUMemory l_CompactedSizeBuffer( sizeof( uint64_t ) );
        GPUMemory l_TempBuffer( l_BlasBufferSizes.tempSizeInBytes );
        GPUMemory l_OutputBuffer( l_BlasBufferSizes.outputSizeInBytes );

        OptixAccelEmitDesc l_EmitDesc;
        l_EmitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        l_EmitDesc.result = l_CompactedSizeBuffer.RawDevicePtr();

        OPTIX_CHECK( optixAccelBuild( m_RTContext->RTObject, 0, &l_AccelOptions, m_TriangleInput.data(), (int)m_TriangleInput.size(), l_TempBuffer.RawDevicePtr(),
                                      l_TempBuffer.Size(), l_OutputBuffer.RawDevicePtr(), l_OutputBuffer.Size(), &RTObject, &l_EmitDesc, 1 ) );
        CUDA_SYNC_CHECK();

        uint64_t l_CompactedSize = l_CompactedSizeBuffer.Fetch<uint64_t>()[0];

        m_ASBuffer = GPUMemory( l_CompactedSize );
        OPTIX_CHECK( optixAccelCompact( m_RTContext->RTObject, 0, RTObject, m_ASBuffer.RawDevicePtr(), m_ASBuffer.Size(), &RTObject ) );

        CUDA_SYNC_CHECK();

        l_TempBuffer.Dispose();
        l_OutputBuffer.Dispose();
        l_CompactedSizeBuffer.Dispose();
    }

    void OptixTraversableObject::Reset() {}

} // namespace LTSE::Graphics
