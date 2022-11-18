#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>


#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "OptixContext.h"
#include "OptixProgramGroup.h"
#include "OptixShaderBindingTable.h"
#include "Optix7.h"


namespace SE::Graphics
{
    using namespace SE::Core;

    struct OptixPipelineObject
    {
        OptixPipeline RTObject;

        OptixPipelineObject() = default;
        OptixPipelineObject( OptixPipelineLinkOptions a_PipelineLinkOptions, OptixPipelineCompileOptions a_PipelineCompileOptions,
                             std::vector<Ref<OptixProgramGroupObject>> a_ProgramGroups, Ref<OptixDeviceContextObject> a_RTContext );

        void Launch( CUstream stream, CUdeviceptr launchParamsBuffer, size_t launchParamBufferSize, Ref<OptixShaderBindingTableObject> a_SBT,
                     math::uvec3 a_LaunchDimensions );

        ~OptixPipelineObject();

      private:
        Ref<OptixDeviceContextObject> m_RTContext = nullptr;
    };

} // namespace SE::Graphics