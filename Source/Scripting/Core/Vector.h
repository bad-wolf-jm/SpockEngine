#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/Entity/Collection.h"
// #include "Core/Cuda/Texture/TextureData.h"

namespace SE::Core
{
    void open_vector_library( sol::table &scriptingState );
}; // namespace SE::Core