#include "utilities.h"

#include "fmt/args.h"
#include "fmt/core.h"

namespace numlua::core::lib::utilities
{
    size_t translate_relative_position( lua_Integer pos, size_t len )
    {
        if( pos > 0 )
            return (size_t)pos;
        else if( pos == 0 )
            return 1;
        else if( pos < -(lua_Integer)len ) /* inverted comparison */
            return 1;                      /* clip to 1 */
        else
            return len + (size_t)pos + 1;
    }

    size_t get_end_position( lua::state_t *L, int arg, lua_Integer def, size_t len )
    {
        lua_Integer pos = luaL_optinteger( L, arg, def );
        if( pos > (lua_Integer)len )
            return len;
        else if( pos >= 0 )
            return (size_t)pos;
        else if( pos < -(lua_Integer)len )
            return 0;
        else
            return len + (size_t)pos + 1;
    }

    int format( lua::state_t *L )
    {
        const char *format = luaL_checkstring( L, 1 );
        int         argc   = lua_gettop( L );

        fmt::dynamic_format_arg_store<fmt::format_context> store;
        for( int i = 2; i <= argc; i++ )
        {
            switch( lua_type( L, i ) )
            {
            case LUA_TNIL:
                store.push_back( "<nil>" );
                break;
            case LUA_TBOOLEAN:
                store.push_back( lua::cast_as<bool>( L, lua::stack_index_t{ i } ) );
                break;
            case LUA_TLIGHTUSERDATA:
                break;
            case LUA_TNUMBER:
                if( lua_isinteger( L, i ) )
                    store.push_back( lua::cast_as<int64_t>( L, lua::stack_index_t{ i } ) );
                else
                    store.push_back( lua::cast_as<double>( L, lua::stack_index_t{ i } ) );
                break;
            case LUA_TSTRING:
                store.push_back( lua::cast_as<const char *>( L, lua::stack_index_t{ i } ) );
                break;
            case LUA_TTABLE:
                break;
            case LUA_TFUNCTION:
                break;
            case LUA_TUSERDATA:
                break;
            case LUA_TTHREAD:
                break;
            }
        }
        std::string result = fmt::vformat( format, store );
        lua::pop( L, argc );
        lua::push( L, result.c_str() );
        return 1;
    }
} // namespace numlua::core::lib::utilities
