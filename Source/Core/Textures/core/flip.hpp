#pragma once

#include <array>

#include "../texture2d.hpp"
#include "../texture2d_array.hpp"
#include "../texture_cube.hpp"
#include "../texture_cube_array.hpp"

namespace numlua::core
{
    template <typename texture>
    texture flip( texture const &Texture );

} // namespace numlua::core

#include "flip.inl"
