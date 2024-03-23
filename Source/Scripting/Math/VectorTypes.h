#pragma once

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace SE::Core
{
    void define_vector_types( sol::table &aModule );
}