#include "lua_context.h"
#include "lua_require.h"

#include "lualib.h"
#include "sol/types.hpp"

#include <type_traits>

namespace numlua::core
{
    // using namespace math;
    using namespace sol;

    lua_context_t::lua_context_t( const char *path, const char *cpath )
    {
        _scriptState.open_libraries( sol::lib::base );
        lib::luaopen_package( _scriptState.lua_state(), path, cpath );
        Initialize();
    }

    void lua_context_t::Initialize()
    {
    }

    environment_t lua_context_t::LoadFile( fs::path path )
    {
        environment_t newEnvironment = NewEnvironment();

        _scriptState.script_file( path.string(), newEnvironment, load_mode::any );

        return newEnvironment;
    }

    environment_t lua_context_t::NewEnvironment()
    {
        environment_t newEnvironment( _scriptState, create, _scriptState.globals() );

        return newEnvironment;
    }

    void lua_context_t::Execute( std::string string )
    {
        _scriptState.script( string );
    }

    void lua_context_t::Execute( environment_t &environment, std::string string )
    {
        _scriptState.script( string, environment );
    }

} // namespace numlua::core
