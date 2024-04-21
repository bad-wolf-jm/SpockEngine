#include "common.h"

namespace numlua::core::details
{
    int do_isabsolute( const char *path )
    {
        // backwards compatibility
        return ( do_absolutetype( path ) == 1 ) ? 1 : 0;
    }

    void do_translate( char *value, const char sep )
    {
        char *ch;
        for( ch = value; *ch != '\0'; ++ch )
        {
            if( *ch == '/' || *ch == '\\' )
            {
                *ch = sep;
            }
        }
    }
    /* Call the scripted path.normalize(), to allow for overrides */
    void do_normalize( lua_State *L, char *buffer, const char *path )
    {
        int top = lua_gettop( L );

        lua_getglobal( L, "path" );
        lua_getfield( L, -1, "normalize" );
        lua_pushstring( L, path );
        lua_call( L, 1, 1 );

        path = luaL_checkstring( L, -1 );
        strcpy( buffer, path );

        lua_settop( L, top );
    }

    int do_absolutetype( const char *path )
    {
        char        c;
        const char *closing;
        size_t      length;

        while( path[0] == '"' || path[0] == '!' )
            path++;
        if( path[0] == '/' || path[0] == '\\' )
            return JOIN_ABSOLUTE;
        if( isalpha( path[0] ) && path[1] == ':' )
            return JOIN_ABSOLUTE;

        // $(foo) and %(foo)
        if( ( path[0] == '%' || path[0] == '$' ) && path[1] == '(' )
        {
            char delimiter = path[0];
            closing        = strchr( path + 2, ')' );
            if( closing == NULL )
                return JOIN_RELATIVE;

            path += 2;
            // special case VS macros %(filename) and %(extension) as normal text
            if( delimiter == '%' )
            {
                length = closing - path;
                switch( length )
                {
                case 8:
                    if( strncasecmp( path, "Filename)", length ) == 0 )
                        return JOIN_RELATIVE;
                    break;
                case 9:
                    if( strncasecmp( path, "Extension)", length ) == 0 )
                        return JOIN_RELATIVE;
                    break;
                default:
                    break;
                }
            }

            // only alpha, digits, _ and . allowed inside $()
            while( path < closing )
            {
                c = *path++;
                if( !isalpha( c ) && !isdigit( c ) && c != '_' && c != '.' )
                    return JOIN_RELATIVE;
            }

            return JOIN_ABSOLUTE;
        }

        // $ORIGIN.
        if( path[0] == '$' )
            return JOIN_ABSOLUTE;

        // either %ORIGIN% or %{<lua code>}
        if( path[0] == '%' )
        {
            if( path[1] == '{' ) //${foo} need to defer join until after detokenization
            {
                closing = strchr( path + 2, '}' );
                if( closing != NULL )
                    return JOIN_MAYBE_ABSOLUTE;
            }
            // find the second closing %
            path += 1;
            closing = strchr( path, '%' );
            if( closing == NULL )
                return JOIN_RELATIVE;

            // need at least one character between the %%
            if( path == closing )
                return JOIN_RELATIVE;

            // only alpha, digits and _ allowed inside %..%
            while( path < closing )
            {
                c = *path++;
                if( !isalpha( c ) && !isdigit( c ) && c != '_' )
                    return JOIN_RELATIVE;
            }
            return JOIN_ABSOLUTE;
        }

        return JOIN_RELATIVE;
    }

    int do_getcwd( char *buffer, size_t size )
        {
            int result;

            wchar_t wbuffer[PATH_MAX];

            result = ( GetCurrentDirectoryW( PATH_MAX, wbuffer ) != 0 );
            if( result )
            {
                WideCharToMultiByte( CP_UTF8, 0, wbuffer, -1, buffer, (int)size, NULL, NULL );

                do_translate( buffer, '/' );
            }

            return result;
        }
} // namespace numlua::core::details