#include "path.h"
#include "common.h"

namespace numlua::core
{
    namespace details
    {
        void do_getabsolute( char *result, const char *value, const char *relative_to )
        {
            int   i;
            char *ch;
            char *prev;
            char  buffer[0x4000] = { '\0' };

            /* if the path is not already absolute, base it on working dir */
            if( !do_isabsolute( value ) )
            {
                if( relative_to )
                {
                    strcpy( buffer, relative_to );
                }
                else
                {
                    do_getcwd( buffer, 0x4000 );
                }
                strcat( buffer, "/" );
            }

            /* normalize the path */
            strcat( buffer, value );
            do_translate( buffer, '/' );

            /* process it part by part */
            result[0] = '\0';
            if( buffer[0] == '/' )
            {
                strcat( result, "/" );
                if( buffer[1] == '/' )
                {
                    strcat( result, "/" );
                }
            }

            prev = NULL;
            ch   = strtok( buffer, "/" );
            while( ch )
            {
                /* remove ".." where I can */
                if( strcmp( ch, ".." ) == 0 && ( prev == NULL || ( prev[0] != '$' && prev[0] != '%' && strcmp( prev, ".." ) != 0 ) ) )
                {
                    i = (int)strlen( result ) - 2;
                    while( i >= 0 && result[i] != '/' )
                    {
                        --i;
                    }
                    if( i >= 0 )
                    {
                        result[i + 1] = '\0';
                    }
                    ch = NULL;
                }

                /* allow everything except "." */
                else if( strcmp( ch, "." ) != 0 )
                {
                    strcat( result, ch );
                    strcat( result, "/" );
                }

                prev = ch;
                ch   = strtok( NULL, "/" );
            }

            /* remove trailing slash */
            i = (int)strlen( result ) - 1;
            if( result[i] == '/' )
            {
                result[i] = '\0';
            }
        }

        int path_getabsolute( lua_State *L )
        {
            const char *relative_to;
            char        buffer[0x4000];

            relative_to = NULL;
            if( lua_gettop( L ) > 1 && !lua_isnil( L, 2 ) )
            {
                relative_to = luaL_checkstring( L, 2 );
            }

            if( lua_istable( L, 1 ) )
            {
                int i = 0;
                lua_newtable( L );
                lua_pushnil( L );
                while( lua_next( L, 1 ) )
                {
                    const char *value = luaL_checkstring( L, -1 );
                    do_getabsolute( buffer, value, relative_to );
                    lua_pop( L, 1 );

                    lua_pushstring( L, buffer );
                    lua_rawseti( L, -3, ++i );
                }
                return 1;
            }
            else
            {
                const char *value = luaL_checkstring( L, 1 );
                do_getabsolute( buffer, value, relative_to );
                lua_pushstring( L, buffer );
                return 1;
            }
        }

        int path_getrelative( lua_State *L )
        {
            int  i, last, count;
            char src[0x4000];
            char dst[0x4000];

            const char *p1 = luaL_checkstring( L, 1 );
            const char *p2 = luaL_checkstring( L, 2 );

            /* normalize the paths */
            do_normalize( L, src, p1 );
            do_normalize( L, dst, p2 );

            /* same directory? */

            if( _stricmp( src, dst ) == 0 )
            {

                lua_pushstring( L, "." );
                return 1;
            }

            /* dollar macro? Can't tell what the real path might be, so treat
             * as absolute. This enables paths like $(SDK_ROOT)/include to
             * work as expected. */
            if( dst[0] == '$' )
            {
                lua_pushstring( L, dst );
                return 1;
            }

            /* find the common leading directories */
            strcat( src, "/" );
            strcat( dst, "/" );

            last = -1;
            i    = 0;

            while( src[i] && dst[i] && tolower( src[i] ) == tolower( dst[i] ) )
            {

                if( src[i] == '/' )
                {
                    last = i;
                }
                ++i;
            }

            /* if I end up with just the root of the filesystem, either a single
             * slash (/) or a drive letter (c:) then return the absolute path. */
            if( last <= 0 || ( last == 2 && src[1] == ':' ) )
            {
                dst[strlen( dst ) - 1] = '\0';
                lua_pushstring( L, dst );
                return 1;
            }

            /* Relative paths within a server can't climb outside the server root.
             * If the paths don't share server name, return the absolute path. */
            if( src[0] == '/' && src[1] == '/' && last == 1 )
            {
                dst[strlen( dst ) - 1] = '\0';
                lua_pushstring( L, dst );
                return 1;
            }

            /* count remaining levels in src */
            count = 0;
            for( i = last + 1; src[i] != '\0'; ++i )
            {
                if( src[i] == '/' )
                {
                    ++count;
                }
            }

            /* start my result by backing out that many levels */
            src[0] = '\0';
            for( i = 0; i < count; ++i )
            {
                strcat( src, "../" );
            }

            /* append what's left */
            strcat( src, dst + last + 1 );

            /* remove trailing slash and done */
            src[strlen( src ) - 1] = '\0';
            lua_pushstring( L, src );
            return 1;
        }

        int do_isabsolute( const char *path )
        {
            // backwards compatibility
            return ( do_absolutetype( path ) == 1 ) ? 1 : 0;
        }

        int path_isabsolute( lua_State *L )
        {
            const char *path = luaL_checkstring( L, -1 );
            lua_pushboolean( L, do_isabsolute( path ) );
            return 1;
        }

        int path_absolutetype( lua_State *L )
        {
            const char *path = luaL_checkstring( L, -1 );
            lua_pushinteger( L, do_absolutetype( path ) );
            return 1;
        }

        char *path_join_single( char *buffer, char *ptr, const char *part, int allowDeferredJoin )
        {
            int    absoluteType;
            size_t len = strlen( part );
            /* remove leading "./" */
            while( strncmp( part, "./", 2 ) == 0 )
            {
                part += 2;
                len -= 2;
            }

            /* remove trailing slashes */
            while( len > 1 && part[len - 1] == '/' )
            {
                --len;
            }

            /* ignore empty segments and "." */
            if( len == 0 || ( len == 1 && part[0] == '.' ) )
            {
                return ptr;
            }

            absoluteType = do_absolutetype( part );
            if( !allowDeferredJoin && absoluteType == JOIN_MAYBE_ABSOLUTE )
                absoluteType = JOIN_RELATIVE;

            /* if I encounter an absolute path, restart my result */
            switch( absoluteType )
            {
            case JOIN_ABSOLUTE:
                ptr = buffer;
                break;
            case JOIN_RELATIVE:
                /* if source has a .. prefix then take off last dest path part
                note that this doesn't guarantee a normalized result as this
                code doesn't check for .. in the mid path, however .. occurring
                mid path are much more likely to occur during path joins
                and its faster if we handle here as we don't have to remove
                substrings from the middle of the string. */

                while( ptr != buffer && len >= 2 && part[0] == '.' && part[1] == '.' )
                {
                    /* locate start of previous segment */
                    char *start = strrchr( buffer, '/' );
                    if( !start )
                    {
                        start = buffer;
                    }
                    else
                    {
                        ++start;
                    }

                    /* if I hit a segment I can't trim, bail out */
                    if( strcmp( start, ".." ) == 0        /* parent dir */
                        || strcmp( start, "." ) == 0      /* current dir */
                        || strstr( start, "**" ) != NULL  /* recursive wildcard */
                        || strchr( start, '$' ) != NULL ) /* property expansion */
                    {
                        break;
                    }

                    /* otherwise trim segment and the ".." sequence */
                    if( start != buffer )
                    {
                        --start;
                    }
                    *start = '\0';
                    ptr    = start;
                    part += 2;
                    len -= 2;
                    if( len > 0 && part[0] == '/' )
                    {
                        ++part;
                        --len;
                    }
                }

                /* if the path is already started, split parts */
                if( ptr != buffer && *( ptr - 1 ) != '/' )
                {
                    *( ptr++ ) = '/';
                }

                break;
            case JOIN_MAYBE_ABSOLUTE:
                *ptr = DEFERRED_JOIN_DELIMITER;
                ptr++;
                break;
            }

            /* append new part */
            strncpy( ptr, part, len );
            ptr += len;
            *ptr = '\0';
            return ptr;
        }

        int path_join_internal( lua_State *L, int allowDeferredJoin )
        {
            int         i;
            const char *part;
            char        buffer[0x4000];
            char       *ptr = buffer;

            /* for each argument... */
            int argc = lua_gettop( L );
            for( i = 1; i <= argc; ++i )
            {
                /* if next argument is nil, skip it */
                if( lua_isnil( L, i ) )
                {
                    continue;
                }

                /* grab the next argument */
                part = luaL_checkstring( L, i );
                ptr  = path_join_single( buffer, ptr, part, allowDeferredJoin );
            }

            lua_pushstring( L, buffer );
            return 1;
        }

        int path_join( lua_State *L )
        {
            return path_join_internal( L, 0 );
        }

        int path_deferred_join( lua_State *L )
        {
            return path_join_internal( L, 1 );
        }

        int do_path_has_deferred_join( const char *path )
        {
            return ( strchr( path, DEFERRED_JOIN_DELIMITER ) != NULL );
        }

        int path_has_deferred_join( lua_State *L )
        {
            const char *path = luaL_checkstring( L, -1 );
            lua_pushboolean( L, do_path_has_deferred_join( path ) );
            return 1;
        }

        int path_resolve_deferred_join( lua_State *L )
        {
            const char *path = luaL_checkstring( L, -1 );
            char        inBuffer[0x4000];
            char        outBuffer[0x4000];
            char       *ptr = outBuffer;
            char       *nextPart;
            size_t      len = strlen( path );
            int         i;
            int         numParts = 0;
            strncpy( inBuffer, path, len );
            inBuffer[len] = '\0';
            char *parts[0x200];
            // break up the string into parts and index the start of each part
            nextPart = strchr( inBuffer, DEFERRED_JOIN_DELIMITER );
            if( nextPart == NULL ) // nothing to do
            {
                lua_pushlstring( L, inBuffer, len );
                return 1;
            }
            parts[numParts++] = inBuffer;
            while( nextPart != NULL )
            {
                *nextPart = '\0';
                nextPart++;
                parts[numParts++] = nextPart;
                nextPart          = strchr( nextPart, DEFERRED_JOIN_DELIMITER );
            }

            /* for each part... */
            for( i = 0; i < numParts; ++i )
            {
                nextPart = parts[i];
                ptr      = path_join_single( outBuffer, ptr, nextPart, 0 );
            }

            lua_pushstring( L, outBuffer );
            return 1;
        }

#define IS_SEP( __c )   ( ( __c ) == '/' || ( __c ) == '\\' )
#define IS_QUOTE( __c ) ( ( __c ) == '\"' || ( __c ) == '\'' )

#define IS_UPPER_ALPHA( __c ) ( ( __c ) >= 'A' && ( __c ) <= 'Z' )
#define IS_LOWER_ALPHA( __c ) ( ( __c ) >= 'a' && ( __c ) <= 'z' )
#define IS_ALPHA( __c )       ( IS_UPPER_ALPHA( __c ) || IS_LOWER_ALPHA( __c ) )

#define IS_SPACE( __c ) ( ( __c >= '\t' && __c <= '\r' ) || __c == ' ' )

#define IS_WIN_ENVVAR_START( __c ) ( *__c == '%' )
#define IS_WIN_ENVVAR_END( __c )   ( *__c == '%' )

#define IS_VS_VAR_START( __c ) ( *__c == '$' && __c[1] == '(' )
#define IS_VS_VAR_END( __c )   ( *__c == ')' )

#define IS_UNIX_ENVVAR_START( __c ) ( *__c == '$' && __c[1] == '{' )
#define IS_UNIX_ENVVAR_END( __c )   ( *__c == '}' )

#define IS_PREMAKE_TOKEN_START( __c ) ( *__c == '%' && __c[1] == '{' )
#define IS_PREMAKE_TOKEN_END( __c )   ( *__c == '}' )

        static void *normalize_substring( const char *srcPtr, const char *srcEnd, char *dstPtr )
        {
#define IS_END( __p )        ( __p >= srcEnd || *__p == '\0' )
#define IS_SEP_OR_END( __p ) ( IS_END( __p ) || IS_SEP( *__p ) )

            // Handle Windows absolute paths
            if( IS_ALPHA( srcPtr[0] ) && srcPtr[1] == ':' )
            {
                *( dstPtr++ ) = srcPtr[0];
                *( dstPtr++ ) = ':';

                srcPtr += 2;
            }

            // Handle path starting with a sep (C:/ or /)
            if( IS_SEP( *srcPtr ) )
            {
                ++srcPtr;
                *( dstPtr++ ) = '/';
                // Handle path starting with //
                if( IS_SEP( *srcPtr ) )
                {
                    ++srcPtr;
                    *( dstPtr++ ) = '/';
                }
            }

            const char *const dstRoot     = dstPtr;
            unsigned int      folderDepth = 0;

            while( !IS_END( srcPtr ) )
            {
                // Skip multiple sep and "./" pattern
                while( IS_SEP( *srcPtr ) || ( srcPtr[0] == '.' && IS_SEP_OR_END( &srcPtr[1] ) ) )
                    ++srcPtr;

                if( IS_END( srcPtr ) )
                    break;

                // Handle "../ pattern"
                if( srcPtr[0] == '.' && srcPtr[1] == '.' && IS_SEP_OR_END( &srcPtr[2] ) )
                {
                    if( folderDepth > 0 )
                    {
                        // Here dstPtr[-1] is safe as folderDepth > 0.
                        while( --dstPtr != dstRoot && !IS_SEP( dstPtr[-1] ) )
                            ;

                        --folderDepth;
                    }
                    else
                    {
                        *( dstPtr++ ) = '.';
                        *( dstPtr++ ) = '.';
                        *( dstPtr++ ) = '/';
                    }
                    srcPtr += 3;
                }
                else
                {
                    while( !IS_SEP_OR_END( srcPtr ) )
                        *( dstPtr++ ) = *( srcPtr++ );

                    if( IS_SEP( *srcPtr ) )
                    {
                        *( dstPtr++ ) = '/';
                        ++srcPtr;
                        ++folderDepth;
                    }
                }
            }

            // Remove trailing slash except for C:/ or / (root)
            while( dstPtr != dstRoot && IS_SEP( dstPtr[-1] ) )
                --dstPtr;

            return dstPtr;
#undef IS_END
#undef IS_SEP_OR_END
        }

        static int skip_tokens( const char *readPtr )
        {
            int skipped = 0;

#define DO_SKIP_FOR( __kind )                       \
    if( IS_##__kind##_START( readPtr ) )            \
    {                                               \
        do                                          \
        {                                           \
            skipped++;                              \
        } while( !IS_##__kind##_END( readPtr++ ) ); \
    }                                               \
    // DO_SKIP_FOR

            do
            {
                DO_SKIP_FOR( PREMAKE_TOKEN )
                DO_SKIP_FOR( WIN_ENVVAR )
                DO_SKIP_FOR( VS_VAR )
                DO_SKIP_FOR( UNIX_ENVVAR )

            } while( IS_WIN_ENVVAR_START( readPtr ) || IS_VS_VAR_START( readPtr ) || IS_UNIX_ENVVAR_START( readPtr ) ||
                     IS_PREMAKE_TOKEN_START( readPtr ) );

            return skipped;
#undef DO_SKIP_FOR
        }

        int path_normalize( lua_State *L )
        {
            const char *path           = luaL_checkstring( L, 1 );
            const char *readPtr        = path;
            char        buffer[0x4000] = { 0 };
            char       *writePtr       = buffer;
            const char *endPtr;

            // skip leading white spaces
            while( IS_SPACE( *readPtr ) )
                ++readPtr;

            endPtr = readPtr;

            while( *endPtr )
            {

                int skipped = skip_tokens( readPtr );
                if( skipped > 0 )
                {

                    if( readPtr != path && writePtr != buffer && IS_SEP( readPtr[-1] ) && !IS_SEP( writePtr[-1] ) )
                    {
                        *( writePtr++ ) = ( readPtr[-1] );
                    }

                    while( skipped-- > 0 )
                        *( writePtr++ ) = *( readPtr++ );

                    endPtr = readPtr;
                }

                // find the end of sub path
                while( *endPtr && !IS_SPACE( *endPtr ) && !IS_WIN_ENVVAR_START( endPtr ) && !IS_VS_VAR_START( endPtr ) &&
                       !IS_UNIX_ENVVAR_START( endPtr ) && !IS_PREMAKE_TOKEN_START( endPtr ) )
                {
                    ++endPtr;
                }

                // path is surrounded with quotes
                if( readPtr != endPtr && IS_QUOTE( *readPtr ) )
                {
                    *( writePtr++ ) = *( readPtr++ );
                }

                writePtr = normalize_substring( readPtr, endPtr, writePtr );

                // skip any white spaces between sub paths
                while( IS_SPACE( *endPtr ) )
                    *( writePtr++ ) = *( endPtr++ );

                readPtr = endPtr;
            }

            // skip any trailing white spaces
            while( writePtr != buffer && IS_SPACE( writePtr[-1] ) )
                --writePtr;

            *writePtr = 0;

            lua_pushstring( L, buffer );
            return 1;
        }

        static void translate( char *result, const char *value, const char sep )
        {
            strcpy( result, value );
            do_translate( result, sep );
        }

        int path_translate( lua_State *L )
        {
            const char *sep;
            char        buffer[0x4000];

            if( lua_gettop( L ) == 1 )
            {
                lua_getglobal( L, "path" );
                lua_getfield( L, -1, "getDefaultSeparator" );
                lua_call( L, 0, 1 );
                sep = luaL_checkstring( L, -1 );
                lua_pop( L, 2 );
            }
            else
            {
                sep = luaL_checkstring( L, 2 );
            }

            if( lua_istable( L, 1 ) )
            {
                int i = 0;
                lua_newtable( L );
                lua_pushnil( L );
                while( lua_next( L, 1 ) )
                {
                    const char *value = luaL_checkstring( L, 4 );
                    translate( buffer, value, sep[0] );
                    lua_pop( L, 1 );

                    lua_pushstring( L, buffer );
                    lua_rawseti( L, -3, ++i );
                }
                return 1;
            }
            else
            {
                const char *value = luaL_checkstring( L, 1 );
                translate( buffer, value, sep[0] );
                lua_pushstring( L, buffer );
                return 1;
            }
        }

        /*
        --Converts from a simple wildcard syntax, where * is "match any"
        -- and ** is "match recursive", to the corresponding Lua pattern.
        --
        -- @param pattern
        --    The wildcard pattern to convert.
        -- @returns
        --    The corresponding Lua pattern.
        */
        int path_wildcards( lua_State *L )
        {
            size_t      length, i;
            const char *input;
            char        buffer[0x4000];
            char       *output;

            input  = luaL_checklstring( L, 1, &length );
            output = buffer;

            for( i = 0; i < length; ++i )
            {
                char c = input[i];
                switch( c )
                {
                case '+':
                case '.':
                case '-':
                case '^':
                case '$':
                case '(':
                case ')':
                case '%':
                    *( output++ ) = '%';
                    *( output++ ) = c;
                    break;

                case '*':
                    if( ( i + 1 ) < length && input[i + 1] == '*' )
                    {
                        i++; // skip the next character.
                        *( output++ ) = '.';
                        *( output++ ) = '*';
                    }
                    else
                    {
                        *( output++ ) = '[';
                        *( output++ ) = '^';
                        *( output++ ) = '/';
                        *( output++ ) = ']';
                        *( output++ ) = '*';
                    }
                    break;

                default:
                    *( output++ ) = c;
                    break;
                }

                if( output >= buffer + sizeof( buffer ) )
                {
                    lua_pushstring( L, "Wildcards expansion too big." );
                    lua_error( L );
                }
            }

            *( output++ ) = '\0';

            lua_pushstring( L, buffer );
            return 1;
        }
    } // namespace details

    void open_path_library( sol::table &aScriptingState )
    {
    }
} // namespace numlua::core
