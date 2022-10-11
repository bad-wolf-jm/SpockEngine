#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/EntityRegistry/Registry.h"
#include "Core/TextureData.h"

namespace LTSE::Core
{
    TextureData::sCreateInfo ParseCreateInfo( sol::table aTable );
    sImageData ParseImageData( sol::table aTable );
    sTextureSamplingInfo ParseSamplerInfo( sol::table aTable );

    void OpenCoreLibrary( sol::table &aScriptingState );
}; // namespace LTSE::Core