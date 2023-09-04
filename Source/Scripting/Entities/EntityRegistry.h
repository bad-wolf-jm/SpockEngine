#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/Entity/Collection.h"

namespace SE::Core
{
    void OpenEntityRegistry( sol::state &aScriptingState );
}; // namespace SE::Core