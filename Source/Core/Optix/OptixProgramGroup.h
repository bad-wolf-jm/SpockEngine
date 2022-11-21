#pragma once

#include <string>
#include <vector>

#include "Core/Memory.h"
#include "Optix7.h"
#include "OptixContext.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    struct OptixProgramGroupObject
    {
        OptixProgramGroup RTObject = nullptr;

        OptixProgramGroupObject() = default;
        OptixProgramGroupObject( OptixProgramGroupDesc a_ProgramGroupDescription, OptixProgramGroupOptions a_ProgramGroupOptions,
                                 Ref<OptixDeviceContextObject> a_RTContext );

        ~OptixProgramGroupObject();

      private:
        Ref<OptixDeviceContextObject> m_RTContext = nullptr;
    };

} // namespace SE::Graphics