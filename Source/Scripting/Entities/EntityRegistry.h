#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/EntityRegistry/Registry.h"

namespace SE::Core
{
    void OpenEntityRegistry( sol::table &aScriptingState );
}; // namespace SE::Core