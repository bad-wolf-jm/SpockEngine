#pragma once

#include <string>
#include <vector>

#include "OptixContext.h"
#include "Optix7.h"
#include "Core/Memory.h"


namespace LTSE::Graphics
{
    using namespace LTSE::Core;

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

} // namespace LTSE::Graphics