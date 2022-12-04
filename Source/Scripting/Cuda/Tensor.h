#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/EntityRegistry/Registry.h"

#include "Core/GPUResource/Array/MultiTensor.h"

namespace SE::Core
{
    void OpenTensorLibrary( sol::table &aScriptingState );
}; // namespace SE::Core