#include "OptixPipeline.h"

#include "Core/CUDA/CudaAssert.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{

    OptixPipelineObject::OptixPipelineObject( OptixPipelineLinkOptions                  aPipelineLinkOptions,
                                              OptixPipelineCompileOptions               aPipelineCompileOptions,
                                              vec_t<ref_t<OptixProgramGroupObject>> aProgramGroups,
                                              ref_t<OptixDeviceContextObject>             aRayTracingContext )
        : mRayTracingContext{ aRayTracingContext }
    {
        vec_t<OptixProgramGroup> lProgramGroups;
        for( auto pg : aProgramGroups ) lProgramGroups.push_back( pg->mOptixObject );

        char   lLogString[2048];
        size_t lLogStringSize = sizeof( lLogString );

        OPTIX_CHECK( optixPipelineCreate( mRayTracingContext->mOptixObject, &aPipelineCompileOptions, &aPipelineLinkOptions,
                                          lProgramGroups.data(), (int)lProgramGroups.size(), lLogString, &lLogStringSize,
                                          &mOptixObject ) );

        if( lLogStringSize > 1 ) SE::Logging::Info( "{}", lLogString );

        OPTIX_CHECK( optixPipelineSetStackSize( mOptixObject, 2 * 1024, 2 * 1024, 2 * 1024, 1 ) );
    }

    OptixPipelineObject::~OptixPipelineObject() { OPTIX_CHECK_NO_EXCEPT( optixPipelineDestroy( mOptixObject ) ); }

    void OptixPipelineObject::Launch( CUstream aStream, RawPointer aLaunchParamsBuffer, size_t aLaunchParamBufferSize,
                                      ref_t<OptixShaderBindingTableObject> aShaderBindingTable, math::uvec3 aLaunchDimensions )
    {
        OPTIX_CHECK( optixLaunch( mOptixObject, aStream, aLaunchParamsBuffer, aLaunchParamBufferSize,
                                  &aShaderBindingTable->mOptixObject, aLaunchDimensions.x, aLaunchDimensions.y,
                                  aLaunchDimensions.z ) );
        SyncDevice();
    }

} // namespace SE::Graphics