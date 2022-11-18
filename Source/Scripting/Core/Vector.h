#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/EntityRegistry/Registry.h"
#include "Core/Textures/TextureData.h"

namespace SE::Core
{
    void OpenVectorLibrary( sol::table &aScriptingState );
}; // namespace SE::Core