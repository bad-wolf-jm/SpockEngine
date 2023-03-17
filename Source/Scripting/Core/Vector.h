#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/EntityRegistry/Registry.h"
#include "Core/TextureData.h"

namespace LTSE::Core
{
    void OpenVectorLibrary( sol::table &aScriptingState );
}; // namespace LTSE::Core