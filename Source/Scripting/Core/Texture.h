#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/Entity/Collection.h"

namespace SE::Core
{
    image_data_t            parse_image_data( sol::table table );
    texture_create_info_t   parse_create_info( sol::table table );
    texture_sampling_info_t parse_sampler_info( sol::table table );

    void open_core_library( sol::table &aScriptingState );
}; // namespace SE::Core