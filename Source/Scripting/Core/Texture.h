#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/Entity/Collection.h"
#include "Core/CUDA/Texture/TextureTypes.h"

namespace SE::Core
{
    texture_create_info_t ParseCreateInfo( sol::table aTable );
    image_data_t ParseImageData( sol::table aTable );
    texture_sampling_info_t ParseSamplerInfo( sol::table aTable );

    void OpenCoreLibrary( sol::table &aScriptingState );
}; // namespace SE::Core