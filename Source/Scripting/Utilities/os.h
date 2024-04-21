#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

// #include "Core/CUDA/Texture/TextureTypes.h"
// #include "Core/Entity/Collection.h"

namespace numlua::core
{
    void open_os_library( sol::table &aScriptingState );
}; // namespace SE::Core