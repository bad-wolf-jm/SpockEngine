#include "OptixPipeline.h"

#include "Core/GPUResource/CudaAssert.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{

    OptixPipelineObject::OptixPipelineObject( OptixPipelineLinkOptions                  aPipelineLinkOptions,
                                              OptixPipelineCompileOptions               aPipelineCompileOptions,
                                              std::vector<Ref<OptixProgramGroupObject>> aProgramGroups,
                                              Ref<OptixDeviceContextObject>             aRayTracingContext )
        : mRayTracingContext{ aRayTracingContext }
    {
        std::vector<OptixProgramGroup> lProgramGroups;
        for( auto pg : aProgramGroups ) lProgramGroups.push_back( pg->mOptixObject );

        char   lLogString[2048];
        size_t lLogStringSize = sizeof( lLogString );

        OPTIX_CHECK( optixPipelineCreate( mRayTracingContext->mOptixObject, &aPipelineCompileOptions, &aPipelineLinkOptions,
                                          lProgramGroups.data(), (int)lProgramGroups.size(), lLogString, &lLogStringSize,
                                          &mOptixObject ) );

        if( lLogStringSize > 1 ) SE::Logging::Info( "{}", lLogString );
        
        OPTIX_CHECK( optixPipelineSetStackSize( mOptixObject, 2 * 1024, 2 * 1024, 2 * 1024, 1 ) );
    }

    OptixPipelineObject::~OptixPipelineObject() { OPTIX_CHECK( optixPipelineDestroy( mOptixObject ) ); }

    void OptixPipelineObject::Launch( CUstream aStream, CUdeviceptr aLaunchParamsBuffer, size_t aLaunchParamBufferSize,
                                      Ref<OptixShaderBindingTableObject> aShaderBindingTable, math::uvec3 aLaunchDimensions )
    {
        OPTIX_CHECK( optixLaunch( mOptixObject, aStream, aLaunchParamsBuffer, aLaunchParamBufferSize,
                                  &aShaderBindingTable->mOptixObject, aLaunchDimensions.x, aLaunchDimensions.y,
                                  aLaunchDimensions.z ) );
        CUDA_SYNC_CHECK();
    }

} // namespace SE::Graphics