#include "logging.h"
#include "lauxlib.h"
#include "utilities.h"
#include <spdlog/spdlog.h>

namespace numlua::core::lib
{
    namespace details
    {
        std::string_view do_format(lua::state_t *L)
        { 
            int argc = lua_gettop( L );
            lua::push( L, utilities::format );
            for( int i = 1; i <= argc; i++ )
                lua::push( L, lua::stack_index_t{ i } );
            lua::call( L, argc, 1 );
            return lua::cast_as<std::string_view>( L, lua::stack_index_t{ -1 } );
        }

        int warn( lua::state_t *L )
        {
            auto message = do_format(L);
            spdlog::warn( message );

            lua::pop(L, 1);
            return 0;
        }

        int critical( lua::state_t *L )
        {
            auto message = do_format(L);
            spdlog::critical( message );

            lua::pop(L, 1);
            return 0;
        }

        int info( lua::state_t *L )
        {
            auto message = do_format(L);
            spdlog::info( message );

            lua::pop(L, 1);
            return 0;
        }

        int error( lua::state_t *L )
        {
            auto message = do_format(L);
            spdlog::error( message );

            lua::pop(L, 1);
            return 0;
        }

        int debug( lua::state_t *L )
        {
            auto message = do_format(L);
            spdlog::debug( message );

            lua::pop(L, 1);
            return 0;
        }

        int set_pattern( lua::state_t *L )
        {
            const char *pattern = luaL_checkstring( L, 1 );
            spdlog::set_pattern( pattern );

            return 0;
        }
    } // namespace details
    static const luaL_Reg functions[] = { { "warn", details::warn },               //
                                          { "critical", details::critical },       //
                                          { "error", details::error },             //
                                          { "info", details::info },               //
                                          { "debug", details::debug },             //
                                          { "set_pattern", details::set_pattern }, //
                                          { NULL, NULL } };

    int open_logging_library( lua::state_t *L )
    {
        luaL_newlib( L, functions );
        return 1;
    }
} // namespace numlua::core::lib
