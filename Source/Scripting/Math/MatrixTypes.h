#pragma once

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace LTSE::Core
{
    void DefineMatrixTypes( sol::table &aModule );
}