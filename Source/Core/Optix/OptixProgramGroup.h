#pragma once

#include <string>
#include <vector>

#include "Core/Memory.h"
#include "Optix7.h"
#include "OptixContext.h"

namespace SE::Graphics
{
    using namespace SE::Core;
    using namespace SE::Cuda;

    struct OptixProgramGroupObject
    {
        OptixProgramGroup mOptixObject = nullptr;

        OptixProgramGroupObject() = default;
        OptixProgramGroupObject( OptixProgramGroupDesc aProgramGroupDescription, OptixProgramGroupOptions aProgramGroupOptions,
                                 ref_t<OptixDeviceContextObject> aRTContext );

        ~OptixProgramGroupObject();

      private:
        ref_t<OptixDeviceContextObject> mRayTracingContext = nullptr;
    };

} // namespace SE::Graphics