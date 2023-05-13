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
                                 Ref<OptixDeviceContextObject> aRTContext );

        ~OptixProgramGroupObject();

      private:
        Ref<OptixDeviceContextObject> mRayTracingContext = nullptr;
    };

} // namespace SE::Graphics