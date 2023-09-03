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
    using namespace SE::Cuda;

    struct OptixPipelineObject
    {
        OptixPipeline mOptixObject;

        OptixPipelineObject() = default;
        OptixPipelineObject( OptixPipelineLinkOptions aPipelineLinkOptions, OptixPipelineCompileOptions aPipelineCompileOptions,
                             vector_t<ref_t<OptixProgramGroupObject>> aProgramGroups, ref_t<OptixDeviceContextObject> aRTContext );

        void Launch( CUstream aStream, RawPointer aLaunchParamsBuffer, size_t aLaunchParamBufferSize,
                     ref_t<OptixShaderBindingTableObject> aShaderBindingTable, math::uvec3 aLaunchDimensions );

        ~OptixPipelineObject();

      private:
        ref_t<OptixDeviceContextObject> mRayTracingContext = nullptr;
    };

} // namespace SE::Graphics