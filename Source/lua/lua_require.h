#pragma once

#include "lua_api.h"

namespace numlua::core::lib
{
    // void find_loader( lua::state_t *L, const char *name );
    // int ll_require( lua::state_t *L );
    int luaopen_package( lua::state_t *L, const char* path, const char* cpath );
}
