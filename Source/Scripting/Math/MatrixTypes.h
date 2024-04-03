#pragma once

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace numlua::core
{
    void define_matrix_types( sol::table &module );
}