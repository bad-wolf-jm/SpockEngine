#pragma once

#include "Optix7.h"

namespace SE::Graphics
{
    struct OptixDeviceContextObject
    {
        OptixDeviceContext mOptixObject = nullptr;

        OptixDeviceContextObject();
        OptixDeviceContextObject( CUcontext aCudaContext );

        ~OptixDeviceContextObject();

        static void Initialize();
    };

} // namespace SE::Graphics