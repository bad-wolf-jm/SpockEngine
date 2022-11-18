#include "OptixPipeline.h"

#include "Core/Logging.h"
#include "Core/Memory.h"
#include "Core/Cuda/CudaAssert.h"


namespace SE::Graphics
{

    OptixPipelineObject::OptixPipelineObject( OptixPipelineLinkOptions a_PipelineLinkOptions, OptixPipelineCompileOptions a_PipelineCompileOptions,
                                              std::vector<Ref<OptixProgramGroupObject>> a_ProgramGroups, Ref<OptixDeviceContextObject> a_RTContext )
        : m_RTContext{ a_RTContext }
    {
        std::vector<OptixProgramGroup> l_ProgramGroups;
        for( auto pg : a_ProgramGroups )
            l_ProgramGroups.push_back( pg->RTObject );

        OPTIX_CHECK( optixPipelineCreate( m_RTContext->RTObject, &a_PipelineCompileOptions, &a_PipelineLinkOptions, l_ProgramGroups.data(), (int)l_ProgramGroups.size(), NULL, NULL,
                                          &RTObject ) );
        OPTIX_CHECK( optixPipelineSetStackSize( RTObject, 2 * 1024, 2 * 1024, 2 * 1024, 1 ) );
    }

    OptixPipelineObject::~OptixPipelineObject() { OPTIX_CHECK( optixPipelineDestroy( RTObject ) ); }

    void OptixPipelineObject::Launch( CUstream stream, CUdeviceptr launchParamsBuffer, size_t launchParamBufferSize, Ref<OptixShaderBindingTableObject> a_SBT,
                                      math::uvec3 a_LaunchDimensions )
    {
        OPTIX_CHECK(
            optixLaunch( RTObject, stream, launchParamsBuffer, launchParamBufferSize, &a_SBT->RTObject, a_LaunchDimensions.x, a_LaunchDimensions.y, a_LaunchDimensions.z ) );
        CUDA_SYNC_CHECK();
    }

} // namespace SE::Graphics