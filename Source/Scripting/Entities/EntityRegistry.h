#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/EntityRegistry/Registry.h"

namespace LTSE::Core
{
    void OpenEntityRegistry( sol::state &aScriptingState );
}; // namespace LTSE::Core