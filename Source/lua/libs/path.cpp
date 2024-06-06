#include "lua/libs/path.h"
#include "lauxlib.h"
#include "lua/lua_api.h"

#include <cctype>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#define JOIN_RELATIVE           0
#define JOIN_ABSOLUTE           1
#define JOIN_MAYBE_ABSOLUTE     2
#define DEFERRED_JOIN_DELIMITER '\a'
#define strncasecmp( x, y, z )  _strnicmp( x, y, z )

namespace numlua::core::lib
{
    namespace details
    {
        int do_isabsolute( std::string_view path )
        {
            if( path.length() < 2 )
                return false;

            auto prefix = path.substr( 0, 2 );
            // backwards compatibility
            return ( isalpha( prefix[0] ) && prefix[1] == ':' ) || prefix == "\\\\";
        }

        int normalize( lua::state_t *L )
        {
            auto p = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );

            std::vector<std::string_view> path;

            size_t offset = 0;
            size_t pos    = p.find_first_of( "\\/" );
            while( pos != std::string::npos )
            {
                auto part = p.substr( offset, pos - offset );

                if( part == "." )
                {
                    offset = p.find_first_not_of( "\\/", pos );
                    pos    = p.find_first_of( "\\/", offset );
                    continue;
                }

                if( part == ".." && path.size() > 0 && path.back() != ".." )
                    path.pop_back();
                else if( part == ".." )
                    path.push_back( part );
                else
                    path.push_back( part );

                offset = p.find_first_not_of( "\\/", pos );
                pos    = p.find_first_of( "\\/", offset );
            }

            auto part = p.substr( offset );
            path.push_back( part );

            std::ostringstream output;
            for( int i = 0; i < path.size() - 1; i++ )
                output << path[i] << "/";
            output << path.back();

            lua::push( L, output.str().c_str() );
            return 1;
        }

        int join( lua::state_t *L )
        {
            std::vector<std::string_view> path;

            int argc = lua_gettop( L );
            for( int i = 1; i <= argc; ++i )
            {
                if( lua::is<lua::nil_t>( L, lua::stack_index_t{ i } ) )
                    continue;

                auto part = lua::cast_as<std::string_view>( L, lua::stack_index_t{ i } );

                if( do_isabsolute( part ) )
                // part[0] == '/' || part[0] == '\\' || ( isalpha( part[0] ) && part[1] == ':' ) )
                {
                    path.clear();
                    path.emplace_back( part );

                    continue;
                }

                path.emplace_back( part );
            }

            std::ostringstream output;
            for( int i = 0; i < path.size() - 1; i++ )
                output << path[i] << "/";
            output << path.back();

            lua::push( L, output.str().c_str() );
            return 1;
        }

        int is_absolute( lua::state_t *L )
        {
            std::string_view path = lua::cast_as<std::string_view>( L, lua::stack_index_t{ -1 } );

            lua::push( L, static_cast<bool>( do_isabsolute( path ) ) );
            return 1;
        }

        void do_normalize( lua::state_t *L, char *buffer, const char *path )
        {
            int top = lua_gettop( L );

            lua::push( L, normalize );
            lua::push( L, path );
            lua_call( L, 1, 1 );

            path = luaL_checkstring( L, -1 );
            strcpy( buffer, path );

            lua_settop( L, top );
        }

        int append_extension( lua::state_t *L )
        {
            auto p = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            if( lua_gettop( L ) < 2 )
            {
                lua::push( L, p.data() );
                return 1;
            }

            std::string result;
            auto        ext = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 2 } );
            result.reserve( p.length() + ext.length() );
            result += p;
            result += ext;
            lua::push( L, result.c_str() );

            return 1;
        }

        int remove_extension( lua::state_t *L )
        {
            auto   p   = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            size_t pos = p.find_last_of( '.' );

            auto result = p.substr( 0, pos );
            lua::push( L, std::string( result.begin(), result.end() ).c_str() );

            return 1;
        }

        int get_extension( lua::state_t *L )
        {
            auto   p   = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            size_t pos = p.find_last_of( '.' );

            auto result = p.substr( pos );
            lua::push( L, std::string( result.begin(), result.end() ).c_str() );

            return 1;
        }

        int get_directory( lua::state_t *L )
        {
            auto   p   = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            size_t pos = p.find_last_of( "\\/" );

            auto result = p.substr( 0, pos );
            lua::push( L, std::string( result.begin(), result.end() ).c_str() );

            return 1;
        }

        int get_name( lua::state_t *L )
        {
            auto   p   = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            size_t pos = p.find_last_of( "\\/" );

            auto result = p.substr( pos );
            lua::push( L, std::string( result.begin() + 1, result.end() ).c_str() );

            return 1;
        }

        int get_base_name( lua::state_t *L )
        {
            auto   p   = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            size_t pos = p.find_last_of( "\\/" );

            auto result = p.substr( pos );
            pos         = result.find_last_of( "." );
            result      = result.substr( 0, pos );
            lua::push( L, std::string( result.begin() + 1, result.end() ).c_str() );

            return 1;
        }

        int has_extension( lua::state_t *L )
        {
            auto   p   = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            size_t pos = p.find_last_of( "." );
            auto   ext = p.substr( pos );

            if( lua::is<lua::table_t>( L, lua::stack_index_t{ 2 } ) )
            {
                lua_pushnil( L );
                while( lua_next( L, -2 ) != 0 )
                {
                    auto testExtension = lua::cast_as<std::string_view>( L, lua::stack_index_t{ -1 } );
                    if( ext == testExtension )
                    {
                        lua::push( L, true );
                        return 1;
                    }

                    lua::pop( L, 1 );
                }

                lua::push( L, false );
                return 1;
            }
            else
            {
                auto testExtension = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 2 } );

                lua::push( L, testExtension == ext );
                return 1;
            }
        }

        int replace_extension( lua::state_t *L )
        {
            auto p      = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            auto newExt = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 2 } );

            size_t pos = p.find_last_of( "." );
            auto   ext = p.substr( pos );
            if( newExt == ext )
            {
                lua::push( L, p.data() );
                return 1;
            }

            std::string result;
            result.reserve( p.length() + newExt.length() );
            result += p.substr( 0, pos );
            result += newExt;

            lua::push( L, result.c_str() );
            return 1;
        }
    } // namespace details

    static const luaL_Reg path_functions[] = { { "is_absolute", details::is_absolute },
                                               { "join", details::join },
                                               { "has_extension", details::has_extension },
                                               { "append_extension", details::append_extension },
                                               { "remove_extension", details::remove_extension },
                                               { "replace_extension", details::replace_extension },
                                               { "get_extension", details::get_extension },
                                               { "get_directory", details::get_directory },
                                               { "get_name", details::get_name },
                                               { "get_base_name", details::get_base_name },
                                               { "normalize", details::normalize },
                                               { NULL, NULL } };

    int open_path_library( lua::state_t *L )
    {
        luaL_newlib( L, path_functions );
        // createmetatable( L );
        return 1;
    }
} // namespace numlua::core::lib
