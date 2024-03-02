#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/Entity/Collection.h"
#include "Core/CUDA/Texture/TextureData.h"

namespace SE::Core
{
    sTextureCreateInfo ParseCreateInfo( sol::table aTable );
    sImageData ParseImageData( sol::table aTable );
    sTextureSamplingInfo ParseSamplerInfo( sol::table aTable );

    void OpenCoreLibrary( sol::table &aScriptingState );
}; // namespace SE::Core