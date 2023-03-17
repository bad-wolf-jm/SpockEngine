#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/EntityRegistry/Registry.h"

#include "Cuda/MultiTensor.h"

namespace LTSE::Core
{
    void OpenTensorLibrary( sol::table &aScriptingState );
}; // namespace LTSE::Core