/*
** $Id: lstrlib.c $
** Standard library for string operations and pattern-matching
** See Copyright Notice in lua.h
*/

// #define lstrlib_c
// #define LUA_LIB

#include "lua/libs/strings.h"
#include "lua/libs/match_state.h"
#include "lua/libs/utilities.h"

#include "fmt/args.h"
#include "fmt/core.h"

#include <algorithm>
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <sstream>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdint>
#include <string_view>

#define MAX_SIZET ( (size_t)( ~(size_t)0 ) )

#define MAXSIZE ( sizeof( size_t ) < sizeof( int ) ? MAX_SIZET : (size_t)( INT_MAX ) )

namespace numlua::core::lib
{
    namespace details
    {
        int len( lua::state_t *L )
        {
            auto str = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );

            lua::push( L, (lua_Integer)str.length() );
            return 1;
        }

        int sub( lua::state_t *L )
        {
            auto str       = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            auto start_pos = lua::cast_as<int64_t>( L, lua::stack_index_t{ 2 } );
            auto end_pos   = lua::cast_as<int64_t>( L, lua::stack_index_t{ 3 }, -1 );

            size_t start = utilities::translate_relative_position( start_pos, str.length() );
            size_t end   = utilities::translate_relative_position( end_pos, str.length() );

            auto substr = str.substr( start - 1, end - start + 1 );

            lua::push( L, &( *substr.begin() ), substr.length() );
            return 1;
        }

        int reverse( lua::state_t *L )
        {
            auto        str  = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            std::string rstr = std::string( str.rbegin(), str.rend() );

            lua::push( L, rstr.c_str() );
            return 1;
        }

        int lower( lua::state_t *L )
        {
            auto        str = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            std::string ustr;
            ustr.resize( str.length() );
            std::transform( str.begin(), str.end(), ustr.begin(), ::tolower );

            lua::push( L, ustr.c_str() );
            return 1;
        }

        int upper( lua::state_t *L )
        {
            auto        str = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            std::string ustr;
            ustr.resize( str.length() );
            std::transform( str.begin(), str.end(), ustr.begin(), ::toupper );

            lua::push( L, ustr.c_str() );
            return 1;
        }

        int rep( lua::state_t *L )
        {
            auto str         = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            auto repetitions = lua::cast_as<int64_t>( L, lua::stack_index_t{ 2 } );
            auto separator   = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 3 }, "" );

            if( repetitions <= 0 )
            {
                lua::push( L, "" );
                return 1;
            }

            size_t strLength   = str.length();
            size_t sepLength   = separator.length();
            size_t totalLength = strLength * repetitions + ( repetitions - 1 ) * sepLength;

            if( strLength + sepLength < strLength || strLength + sepLength > MAXSIZE / repetitions )
                return lua::error( L, "resulting string too large" );

            std::ostringstream p;
            while( repetitions-- > 1 )
            {
                p << str;
                if( sepLength > 0 )
                    p << separator;
            }

            p << str;

            lua::push( L, p.str().c_str() );
            return 1;
        }

        int ends_with( lua::state_t *L )
        {
            std::string_view str    = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            std::string_view suffix = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 2 } );

            bool result = ( str.size() >= suffix.size() ) && ( str.compare( str.size() - suffix.size(), suffix.size(), suffix ) == 0 );

            lua::push( L, result );
            return 1;
        }

        int starts_with( lua::state_t *L )
        {
            std::string_view str    = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            std::string_view prefix = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 2 } );

            bool result = ( str.size() >= prefix.size() ) && ( str.compare( 0, prefix.size(), prefix ) == 0 );

            lua::push( L, result );
            return 1;
        }

        int explode( lua::state_t *L )
        {
            std::string_view str     = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            std::string_view pattern = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 2 } );
            lua_createtable( L, 0, 0 );

            size_t  pos   = 0;
            size_t  start = 0;
            int64_t idx   = 1;
            pos           = str.find( pattern, start );
            while( pos != std ::string::npos )
            {
                auto tok = str.substr( start, pos );
                lua::push( L, tok.data(), pos - start );
                lua::raw_set( L, lua::stack_index_t{ -2 }, idx++ );

                start = pos + pattern.length();
                pos   = str.find( pattern, start );
            }
            auto tok = str.substr( start, pos );
            lua::push( L, tok.data() );
            lua::raw_set( L, lua::stack_index_t{ -2 }, idx++ );

            return 1;
        }

        int contains( lua::state_t *L )
        {
            std::string_view str     = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            std::string_view pattern = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 2 } );
            size_t           pos     = str.find( pattern, 0 );

            lua::push( L, pos != std::string::npos );
            return 1;
        }

        int format( lua::state_t *L )
        {
            return utilities::format( L );
        }

        /* check whether pattern has no special characters */
        static bool nospecials( std::string_view const &p, size_t l )
        {
            auto p1 = p.substr( 0, l );
            return p1.find_first_of( SPECIALS ) == std::string::npos;
        }

        int str_find_aux( lua::state_t *L, int find )
        {
            auto s   = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 1 } );
            auto p   = lua::cast_as<std::string_view>( L, lua::stack_index_t{ 2 } );
            auto pos = lua::cast_as<int64_t>( L, lua::stack_index_t{ 3 }, 1 );

            size_t ls, lp;
            size_t init = utilities::translate_relative_position( pos, s.length() ) - 1;

            if( init > s.length() )
            {                                 /* start after string's end? */
                lua::push( L, lua::nil_t{} ); /* cannot find anything */
                return 1;
            }

            p            = p.substr( init );
            auto isPlain = lua::cast_as<bool>( L, lua::stack_index_t{ 4 } );

            /* explicit request or no special characters? */
            if( find && ( isPlain || nospecials( p, p.length() ) ) )
            {

                auto pos = s.find( p );
                if( pos != std::string::npos )
                {
                    lua::push( L, static_cast<int64_t>( pos + 1 ) );
                    lua::push( L, static_cast<int64_t>( pos + p.length() ) );
                    return 2;
                }
            }
            else
            {
                match_state_t ms;
                const char   *p1   = p.data();
                lp                 = p.length();
                const char *s1     = s.data() + init;
                int         anchor = ( p[0] == '^' );

                if( anchor )
                {
                    p1++;
                    lp--; /* skip anchor character */
                }

                ms.prepstate( L, s.data(), s.length(), p1, lp );

                do
                {
                    const char *res;
                    ms.reprepstate();
                    if( ( res = ms.match( s1, p1 ) ) != NULL )
                    {
                        if( find )
                        {
                            lua::push( L, static_cast<int64_t>( ( s1 - s.data() ) + 1 ) ); /* start */
                            lua::push( L, static_cast<int64_t>( res - s.data() ) );        /* end */
                            return ms.push_captures( NULL, 0 ) + 2;
                        }
                        else
                            return ms.push_captures( s1, res );
                    }
                } while( s1++ < ms.src_end && !anchor );
            }

            lua::push( L, lua::nil_t{} ); /* not found */
            return 1;
        }

        int find( lua::state_t *L )
        {
            return str_find_aux( L, 1 );
        }

        int match( lua::state_t *L )
        {
            return str_find_aux( L, 0 );
        }

        /* state for 'gmatch' */
        struct gmatch_state_t
        {
            const char   *src;       /* current position */
            const char   *p;         /* pattern */
            const char   *lastmatch; /* end of last match */
            match_state_t ms;        /* match state */
        };

        int gmatch_aux( lua::state_t *L )
        {
            gmatch_state_t *gm = (gmatch_state_t *)lua_touserdata( L, lua_upvalueindex( 3 ) );
            const char     *src;
            gm->ms.L = L;
            for( src = gm->src; src <= gm->ms.src_end; src++ )
            {
                const char *e;
                gm->ms.reprepstate();
                if( ( e = gm->ms.match( src, gm->p ) ) != NULL && e != gm->lastmatch )
                {
                    gm->src = gm->lastmatch = e;
                    return gm->ms.push_captures( src, e );
                }
            }
            return 0; /* not found */
        }

        int gmatch( lua::state_t *L )
        {
            size_t          ls, lp;
            const char     *s    = luaL_checklstring( L, 1, &ls );
            const char     *p    = luaL_checklstring( L, 2, &lp );
            size_t          init = utilities::translate_relative_position( luaL_optinteger( L, 3, 1 ), ls ) - 1;
            gmatch_state_t *gm;
            lua_settop( L, 2 ); /* keep strings on closure to avoid being collected */
            gm = (gmatch_state_t *)lua_newuserdatauv( L, sizeof( gmatch_state_t ), 0 );
            if( init > ls )    /* start after string's end? */
                init = ls + 1; /* avoid overflows in 's + init' */
            gm->ms.prepstate( L, s, ls, p, lp );
            gm->src       = s + init;
            gm->p         = p;
            gm->lastmatch = NULL;
            lua_pushcclosure( L, gmatch_aux, 3 );
            return 1;
        }

        int gsub( lua::state_t *L )
        {
            size_t        srcl, lp;
            const char   *src       = luaL_checklstring( L, 1, &srcl );  /* subject */
            const char   *p         = luaL_checklstring( L, 2, &lp );    /* pattern */
            const char   *lastmatch = NULL;                              /* end of last match */
            int           tr        = lua_type( L, 3 );                  /* replacement type */
            lua_Integer   max_s     = luaL_optinteger( L, 4, srcl + 1 ); /* max replacements */
            int           anchor    = ( *p == '^' );
            lua_Integer   n         = 0; /* replacement count */
            int           changed   = 0; /* change flag */
            match_state_t ms;
            luaL_Buffer   b;
            luaL_argexpected( L, tr == LUA_TNUMBER || tr == LUA_TSTRING || tr == LUA_TFUNCTION || tr == LUA_TTABLE, 3,
                              "string/function/table" );
            luaL_buffinit( L, &b );
            if( anchor )
            {
                p++;
                lp--; /* skip anchor character */
            }
            ms.prepstate( L, src, srcl, p, lp );
            while( n < max_s )
            {
                const char *e;
                ms.reprepstate(); /* (re)prepare state for new match */
                if( ( e = ms.match( src, p ) ) != NULL && e != lastmatch )
                {
                    /* match? */
                    n++;
                    changed = ms.add_value( &b, src, e, tr ) | changed;
                    src = lastmatch = e;
                }
                else if( src < ms.src_end ) /* otherwise, skip one character */
                    luaL_addchar( &b, *src++ );
                else
                    break; /* end of subject */
                if( anchor )
                    break;
            }
            if( !changed )                               /* no changes? */
                lua::push( L, lua::stack_index_t{ 1 } ); /* return original string */
            else
            { /* something changed */
                luaL_addlstring( &b, src, ms.src_end - src );
                luaL_pushresult( &b ); /* create and return new string */
            }
            lua::push( L, n ); /* number of substitutions */
            return 2;
        }
    } // namespace details

    /* }====================================================== */
    static const luaL_Reg strlib[] = { { "startswith", details::starts_with }, //
                                       { "find", details::find },              //
                                       { "format", details::format },          //
                                       { "gmatch", details::gmatch },          //
                                       { "gsub", details::gsub },              //
                                       { "len", details::len },                //
                                       { "lower", details::lower },            //
                                       { "match", details::match },            //
                                       { "rep", details::rep },                //
                                       { "reverse", details::reverse },        //
                                       { "sub", details::sub },                //
                                       { "upper", details::upper },            //
                                       { "endswith", details::ends_with },     //
                                       { "explode", details::explode },        //
                                       { "contains", details::contains },      //
                                       //{ "packsize", str_packsize }, //
                                       //{ "unpack", str_unpack },     //
                                       { NULL, NULL } };

    static const luaL_Reg stringmetamethods[] = { { "__index", NULL }, /* placeholder */
                                                  { NULL, NULL } };

    static void createmetatable( lua::state_t *L )
    {
        /* table to be metatable for strings */
        luaL_newlibtable( L, stringmetamethods );
        luaL_setfuncs( L, stringmetamethods, 0 );
        lua_pushliteral( L, "" );                 /* dummy string */
        lua::push( L, lua::stack_index_t{ -2 } ); /* copy table */
        lua_setmetatable( L, -2 );                /* set table as metatable for strings */
        lua_pop( L, 1 );                          /* pop dummy string */
        lua::push( L, lua::stack_index_t{ -2 } ); /* get string library */
        lua_setfield( L, -2, "__index" );         /* metatable.__index = string */
        lua_pop( L, 1 );                          /* pop metatable */
    }

    /*
    ** Open string library
    */
    int luaopen_string( lua::state_t *L )
    {
        luaL_newlib( L, strlib );
        createmetatable( L );
        return 1;
    }

} // namespace numlua::core::lib
