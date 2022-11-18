#pragma once

#include "Optix7.h"

namespace SE::Graphics
{
    struct OptixDeviceContextObject
    {
        OptixDeviceContext RTObject = nullptr;

        OptixDeviceContextObject();
        OptixDeviceContextObject( CUcontext a_CudaContext );

        ~OptixDeviceContextObject();

        static void Initialize();
    };

} // namespace SE::Graphics