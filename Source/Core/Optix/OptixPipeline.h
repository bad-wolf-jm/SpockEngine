#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Optix7.h"
#include "OptixContext.h"
#include "OptixProgramGroup.h"
#include "OptixShaderBindingTable.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    struct OptixPipelineObject
    {
        OptixPipeline mOptixObject;

        OptixPipelineObject() = default;
        OptixPipelineObject( OptixPipelineLinkOptions aPipelineLinkOptions, OptixPipelineCompileOptions aPipelineCompileOptions,
                             std::vector<Ref<OptixProgramGroupObject>> aProgramGroups, Ref<OptixDeviceContextObject> aRTContext );

        void Launch( CUstream aStream, CUdeviceptr aLaunchParamsBuffer, size_t aLaunchParamBufferSize,
                     Ref<OptixShaderBindingTableObject> aShaderBindingTable, math::uvec3 aLaunchDimensions );

        ~OptixPipelineObject();

      private:
        Ref<OptixDeviceContextObject> mRayTracingContext = nullptr;
    };

} // namespace SE::Graphics