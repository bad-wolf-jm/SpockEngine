#pragma once

#include "Optix7.h"

namespace LTSE::Graphics
{
    struct OptixDeviceContextObject
    {
        OptixDeviceContext RTObject = nullptr;

        OptixDeviceContextObject();
        OptixDeviceContextObject( CUcontext a_CudaContext );

        ~OptixDeviceContextObject();

        static void Initialize();
    };

} // namespace LTSE::Graphics