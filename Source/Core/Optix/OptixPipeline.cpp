#include "OptixPipeline.h"

#include "Core/Cuda/CudaAssert.h"
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

        OPTIX_CHECK( optixPipelineCreate( mRayTracingContext->mOptixObject, &aPipelineCompileOptions, &aPipelineLinkOptions,
                                          lProgramGroups.data(), (int)lProgramGroups.size(), NULL, NULL, &mOptixObject ) );
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