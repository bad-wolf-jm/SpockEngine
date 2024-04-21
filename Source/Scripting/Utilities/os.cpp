#include "os.h"

namespace numlua::core
{
    namespace details
    {
        int do_chdir( lua_State *L, const char *path )
        {
            int z;

            wchar_t wide_buffer[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, path, -1, wide_buffer, PATH_MAX ) == 0 )
            {
                lua_pushstring( L, "unable to encode path" );
                return lua_error( L );
            }

            z = SetCurrentDirectoryW( wide_buffer );

            return z;
        }

        int os_chmod( lua_State *L )
        {
            int   rv;
            char *endPtr;

            const char *path    = luaL_checkstring( L, 1 );
            const char *modeStr = luaL_checkstring( L, 2 );

            int mode = (int)strtol( modeStr, &endPtr, 8 );

            /* DOS-mode permissions only support the low word */
            mode = mode & 0x0000ffff;
            rv   = _chmod( path, mode );

            if( rv != 0 )
            {
                lua_pushnil( L );
                lua_pushfstring( L, "unable to set mode %o on '%s', errno %d : %s", mode, path, errno, strerror( errno ) );
                return 2;
            }
            else
            {
                lua_pushboolean( L, 1 );
                return 1;
            }
        }

        int os_comparefiles( lua_State *L )
        {
            FILE       *firstFile;
            FILE       *secondFile;
            size_t      firstSize;
            size_t      secondSize;
            size_t      count;
            size_t      read;
            char        firstBuffer[4096];
            char        secondBuffer[4096];
            const char *firstPath  = luaL_checkstring( L, 1 );
            const char *secondPath = luaL_checkstring( L, 2 );

            wchar_t wide_firstPath[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, firstPath, -1, wide_firstPath, PATH_MAX ) == 0 )
            {
                lua_pushnil( L );
                lua_pushstring( L, "unable to encode first path" );
                return 2;
            }

            wchar_t wide_secondPath[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, secondPath, -1, wide_secondPath, PATH_MAX ) == 0 )
            {
                lua_pushnil( L );
                lua_pushstring( L, "unable to encode second path" );
                return 2;
            }

            firstFile  = _wfopen( wide_firstPath, L"rb" );
            secondFile = _wfopen( wide_secondPath, L"rb" );

            if( !firstFile )
            {
                if( secondFile )
                    fclose( secondFile );

                lua_pushnil( L );
                lua_pushstring( L, "failed to open first file" );
                return 2;
            }

            if( !secondFile )
            {
                fclose( firstFile );

                lua_pushnil( L );
                lua_pushstring( L, "failed to open second file" );
                return 2;
            }

            // check sizes.
            fseek( firstFile, 0, SEEK_END );
            firstSize = ftell( firstFile );
            fseek( firstFile, 0, SEEK_SET );

            fseek( secondFile, 0, SEEK_END );
            secondSize = ftell( secondFile );
            fseek( secondFile, 0, SEEK_SET );

            if( firstSize != secondSize )
            {
                fclose( firstFile );
                fclose( secondFile );

                lua_pushboolean( L, 0 );
                return 1;
            }

            // compare file content
            while( firstSize > 0 )
            {
                count = firstSize > 4096 ? 4096 : firstSize;

                read = fread( firstBuffer, 1, count, firstFile );
                if( read != count )
                {
                    fclose( firstFile );
                    fclose( secondFile );

                    lua_pushnil( L );
                    lua_pushstring( L, "failed to read first file content" );
                    return 2;
                }

                read = fread( secondBuffer, 1, count, secondFile );
                if( read != count )
                {
                    fclose( firstFile );
                    fclose( secondFile );

                    lua_pushnil( L );
                    lua_pushstring( L, "failed to read second file content" );
                    return 2;
                }

                if( memcmp( firstBuffer, secondBuffer, count ) != 0 )
                {
                    fclose( firstFile );
                    fclose( secondFile );

                    lua_pushboolean( L, 0 );
                    return 1;
                }

                firstSize -= count;
            }

            // File content match
            fclose( firstFile );
            fclose( secondFile );

            lua_pushboolean( L, 1 );
            return 1;
        }

        int os_copyfile( lua_State *L )
        {
            int         z;
            const char *src = luaL_checkstring( L, 1 );
            const char *dst = luaL_checkstring( L, 2 );

            wchar_t wide_src[PATH_MAX];
            wchar_t wide_dst[PATH_MAX];

            if( MultiByteToWideChar( CP_UTF8, 0, src, -1, wide_src, PATH_MAX ) == 0 )
            {
                lua_pushstring( L, "unable to encode source path" );
                return lua_error( L );
            }

            if( MultiByteToWideChar( CP_UTF8, 0, dst, -1, wide_dst, PATH_MAX ) == 0 )
            {
                lua_pushstring( L, "unable to encode source path" );
                return lua_error( L );
            }

            z = CopyFileW( wide_src, wide_dst, FALSE );

            if( !z )
            {
                lua_pushnil( L );
                wchar_t buf[256];
                FormatMessageW( FORMAT_MESSAGE_FROM_SYSTEM, NULL, GetLastError(), MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ), buf,
                                256, NULL );

                char bufA[256];
                WideCharToMultiByte( CP_UTF8, 0, buf, 256, bufA, 256, 0, 0 );

                lua_pushfstring( L, "unable to copy file to '%s', reason: '%s'", dst, bufA );
                return 2;
            }
            else
            {
                lua_pushboolean( L, 1 );
                return 1;
            }
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

        int os_getpass( lua_State *L )
        {
            const char *prompt = luaL_checkstring( L, 1 );

            HANDLE      hstdout = GetStdHandle( STD_OUTPUT_HANDLE );
            HANDLE      hstdin  = GetStdHandle( STD_INPUT_HANDLE );
            DWORD       read_chars, mode, written_chars;
            char        buffer[1024];
            const char *newline = "\n";

            WriteConsoleA( hstdout, prompt, (DWORD)strlen( prompt ), &written_chars, NULL );

            GetConsoleMode( hstdin, &mode );
            SetConsoleMode( hstdin, ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT );
            ReadConsoleA( hstdin, buffer, sizeof( buffer ), &read_chars, NULL );
            SetConsoleMode( hstdin, mode );

            WriteConsoleA( hstdout, newline, (DWORD)strlen( newline ), &written_chars, NULL );

            buffer[strcspn( buffer, "\r\n" )] = '\0';

            lua_pushstring( L, buffer );
            return 1;
        }

        int os_isdir( lua_State *L )
        {
            struct stat buf;
            const char *path = luaL_checkstring( L, 1 );
            DWORD       attr;

            wchar_t wide_path[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, path, -1, wide_path, PATH_MAX ) == 0 )
            {
                lua_pushstring( L, "unable to encode path" );
                return lua_error( L );
            }

            /* empty path is equivalent to ".", must be true */
            if( strlen( path ) == 0 )
            {
                lua_pushboolean( L, 1 );
            }
            // Use Windows-specific GetFileAttributes since it deals with symbolic links.
            else if( ( attr = GetFileAttributesW( wide_path ) ) != INVALID_FILE_ATTRIBUTES )
            {
                int isdir = ( attr & FILE_ATTRIBUTE_DIRECTORY ) != 0;
                lua_pushboolean( L, isdir );
            }
            else if( stat( path, &buf ) == 0 )
            {
                int isdir = ( buf.st_mode & S_IFDIR ) != 0;
                lua_pushboolean( L, isdir );
            }
            else
            {
                lua_pushboolean( L, 0 );
            }

            return 1;
        }

        int do_isfile( lua_State *L, const char *filename )
        {
            wchar_t wide_path[PATH_MAX];
            DWORD   attrib;

            if( MultiByteToWideChar( CP_UTF8, 0, filename, -1, wide_path, PATH_MAX ) == 0 )
            {
                lua_pushstring( L, "unable to encode filepath" );
                return lua_error( L );
            }

            attrib = GetFileAttributesW( wide_path );
            if( attrib != INVALID_FILE_ATTRIBUTES )
            {
                return ( attrib & FILE_ATTRIBUTE_DIRECTORY ) == 0;
            }

            return 0;
        }

        int os_islink( lua_State *L )
        {
            const char *path = luaL_checkstring( L, 1 );

            {
                wchar_t wide_path[PATH_MAX];
                DWORD   attr;
                if( MultiByteToWideChar( CP_UTF8, 0, path, -1, wide_path, PATH_MAX ) == 0 )
                {
                    lua_pushstring( L, "unable to encode path" );
                    return lua_error( L );
                }

                attr = GetFileAttributesW( wide_path );
                if( attr != INVALID_FILE_ATTRIBUTES )
                {
                    lua_pushboolean( L, ( attr & FILE_ATTRIBUTE_REPARSE_POINT ) != 0 );
                    return 1;
                }
            }

            lua_pushboolean( L, 0 );
            return 1;
        }

        int do_mkdir( const char *path )
        {
            struct stat sb;
            char        sub_path[1024];
            int         i, length;

            // if it already exists, return.
            if( stat( path, &sb ) == 0 )
                return 1;

            // find the parent folder name.
            length = (int)strlen( path );
            for( i = length - 1; i >= 0; --i )
            {
                if( path[i] == '/' || path[i] == '\\' )
                    break;
            }

            // if we found one, create it.
            if( i > 0 )
            {
                memcpy( sub_path, path, i );
                sub_path[i] = '\0';

                if( sub_path[i - 1] == ':' )
                {
                    sub_path[i + 0] = '/';
                    sub_path[i + 1] = '\0';
                }

                if( !do_mkdir( sub_path ) )
                    return 0;
            }

            // now finally create the actual folder we want.
            return _mkdir( path ) == 0;
        }

        int os_remove( lua_State *L )
        {
            const char *filename = luaL_checkstring( L, 1 );

            wchar_t wide_path[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, filename, -1, wide_path, PATH_MAX ) == 0 )
            {
                lua_pushstring( L, "unable to encode path" );
                return lua_error( L );
            }

            if( DeleteFileW( wide_path ) )
            {
                lua_pushboolean( L, 1 );
                return 1;
            }
            else
            {
                DWORD err = GetLastError();

                char unicodeErr[512];

                LPWSTR messageBuffer = NULL;
                if( FormatMessageW( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
                                    err, MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ), (LPWSTR)&messageBuffer, 0, NULL ) != 0 )
                {
                    if( WideCharToMultiByte( CP_UTF8, 0, messageBuffer, -1, unicodeErr, sizeof( unicodeErr ), NULL, NULL ) == 0 )
                        strcpy( unicodeErr, "failed to translate error message" );

                    LocalFree( messageBuffer );
                }
                else
                    strcpy( unicodeErr, "failed to get error message" );

                lua_pushnil( L );
                lua_pushfstring( L, "%s: %s", filename, unicodeErr );
                lua_pushinteger( L, err );
                return 3;
            }
        }

        int os_rename( lua_State *L )
        {
            const char *fromname = luaL_checkstring( L, 1 );
            const char *toname   = luaL_checkstring( L, 2 );

            wchar_t wide_frompath[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, fromname, -1, wide_frompath, PATH_MAX ) == 0 )
            {
                lua_pushstring( L, "unable to encode source path" );
                return lua_error( L );
            }

            wchar_t wide_topath[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, toname, -1, wide_topath, PATH_MAX ) == 0 )
            {
                lua_pushstring( L, "unable to encode dest path" );
                return lua_error( L );
            }

            if( MoveFileExW( wide_frompath, wide_topath, MOVEFILE_COPY_ALLOWED ) )
            {
                lua_pushboolean( L, 1 );
                return 1;
            }
            else
            {
                DWORD err = GetLastError();

                char unicodeErr[512];

                LPWSTR messageBuffer = NULL;
                if( FormatMessageW( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
                                    err, MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ), (LPWSTR)&messageBuffer, 0, NULL ) != 0 )
                {
                    if( WideCharToMultiByte( CP_UTF8, 0, messageBuffer, -1, unicodeErr, sizeof( unicodeErr ), NULL, NULL ) == 0 )
                        strcpy( unicodeErr, "failed to translate error message" );

                    LocalFree( messageBuffer );
                }
                else
                    strcpy( unicodeErr, "failed to get error message" );

                lua_pushnil( L );
                lua_pushfstring( L, "%s: %s", fromname, unicodeErr );
                lua_pushinteger( L, err );
                return 3;
            }
        }

        int os_rmdir( lua_State *L )
        {
            int         z;
            const char *path = luaL_checkstring( L, 1 );

            wchar_t wide_path[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, path, -1, wide_path, PATH_MAX ) == 0 )
            {
                lua_pushstring( L, "unable to encode path" );
                return lua_error( L );
            }

            z = RemoveDirectoryW( wide_path );

            if( !z )
            {
                lua_pushnil( L );
                lua_pushfstring( L, "unable to remove directory '%s'", path );
                return 2;
            }
            else
            {
                lua_pushboolean( L, 1 );
                return 1;
            }
        }

        int os_stat( lua_State *L )
        {
            const char *filename = luaL_checkstring( L, 1 );

            struct _stat s;

            wchar_t wide_filename[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, filename, -1, wide_filename, PATH_MAX ) == 0 )
            {
                lua_pushstring( L, "unable to encode source path" );
                return lua_error( L );
            }

            if( _wstat( wide_filename, &s ) != 0 )
            {
                lua_pushnil( L );
                switch( errno )
                {
                case EACCES:
                    lua_pushfstring( L, "'%s' could not be accessed", filename );
                    break;
                case ENOENT:
                    lua_pushfstring( L, "'%s' was not found", filename );
                    break;
                default:
                    lua_pushfstring( L, "An  unknown error %d occured while accessing '%s'", errno, filename );
                    break;
                }
                return 2;
            }

            lua_newtable( L );

            lua_pushstring( L, "mtime" );
            lua_pushinteger( L, (lua_Integer)s.st_mtime );
            lua_settable( L, -3 );

            lua_pushstring( L, "size" );
            lua_pushnumber( L, (lua_Number)s.st_size );
            lua_settable( L, -3 );

            return 1;
        }

        static int truncate_file( const char *fn )
        {
            FILE  *file = fopen( fn, "rb" );
            size_t size;
            file = fopen( fn, "ab" );
            if( file == NULL )
            {
                return FALSE;
            }
            fseek( file, 0, SEEK_END );
            size = ftell( file );
            // append a dummy space. There are better ways to do
            // a touch, however this is a rather simple
            // multiplatform method
            if( fwrite( " ", 1, 1, file ) != 1 )
            {
                fclose( file );
                return FALSE;
            }
            if( _chsize( _fileno( file ), (long)size ) != 0 )
            {
                fclose( file );
                return FALSE;
            }
            fclose( file );
            if( truncate( fn, (off_t)size ) != 0 )
            {
                return FALSE;
            }
            return TRUE;
        }

        int os_touchfile( lua_State *L )
        {
            FILE       *file;
            const char *dst = luaL_checkstring( L, 1 );

            // if destination exist, mark the file as modified
            if( do_isfile( L, dst ) )
            {
                SYSTEMTIME systemTime;
                FILETIME   fileTime;
                HANDLE     fileHandle;
                wchar_t    wide_path[PATH_MAX];
                if( MultiByteToWideChar( CP_UTF8, 0, dst, -1, wide_path, PATH_MAX ) == 0 )
                {
                    lua_pushinteger( L, -1 );
                    lua_pushstring( L, "unable to encode path" );
                    return 2;
                }

                fileHandle = CreateFileW( wide_path, FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                                          FILE_ATTRIBUTE_NORMAL, NULL );
                if( fileHandle == NULL )
                {
                    lua_pushinteger( L, -1 );
                    lua_pushfstring( L, "unable to touch file '%s'", dst );
                    return 2;
                }

                GetSystemTime( &systemTime );
                if( SystemTimeToFileTime( &systemTime, &fileTime ) == 0 )
                {
                    lua_pushinteger( L, -1 );
                    lua_pushfstring( L, "unable to touch file '%s'", dst );
                    return 2;
                }

                if( SetFileTime( fileHandle, NULL, NULL, &fileTime ) == 0 )
                {
                    lua_pushinteger( L, -1 );
                    lua_pushfstring( L, "unable to touch file '%s'", dst );
                    return 2;
                }

                lua_pushinteger( L, 0 );
                return 1;
            }

            wchar_t wide_path[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, dst, -1, wide_path, PATH_MAX ) == 0 )
            {
                lua_pushinteger( L, -1 );
                lua_pushstring( L, "unable to encode path" );
                return 2;
            }

            file = _wfopen( wide_path, L"wb" );

            if( file != NULL )
            {
                fclose( file );

                lua_pushinteger( L, 1 );
                return 1;
            }

            lua_pushinteger( L, -1 );
            lua_pushfstring( L, "unable to open file to '%s'", dst );
            return 2;
        }

        /*
         * Pull off the four lowest bytes of a value and add them to my array,
         * without the help of the determinately sized C99 data types that
         * are not yet universally supported.
         */
        static void add( unsigned char *bytes, int offset, uint32_t value )
        {
            int i;
            for( i = 0; i < 4; ++i )
            {
                bytes[offset++] = (unsigned char)( value & 0xff );
                value >>= 8;
            }
        }

        int os_uuid( lua_State *L )
        {
            char          uuid[38];
            unsigned char bytes[16];

            /* If a name argument is supplied, build the UUID from that. For speed we
             * are using a simple DBJ2 hashing function; if this isn't sufficient we
             * can switch to a full RFC 4122 §4.3 implementation later. */
            const char *name = luaL_optstring( L, 1, NULL );
            if( name != NULL )
            {
                add( bytes, 0, do_hash( name, 0 ) );
                add( bytes, 4, do_hash( name, 'L' ) );
                add( bytes, 8, do_hash( name, 'u' ) );
                add( bytes, 12, do_hash( name, 'a' ) );
            }

            /* If no name is supplied, try to build one properly */
            else
            {
                CoCreateGuid( (GUID *)bytes );
            }

            sprintf( uuid, "%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X", bytes[0], bytes[1], bytes[2],
                     bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13],
                     bytes[14], bytes[15] );

            lua_pushstring( L, uuid );
            return 1;
        }

        struct OsVersionInfo
        {
            int         majorversion;
            int         minorversion;
            int         revision;
            const char *description;
            int         isalloc;
        };

        static int getversion( struct OsVersionInfo *info );

        int os_getversion( lua_State *L )
        {
            struct OsVersionInfo info = { 0, 0, 0, NULL, 0 };
            if( !getversion( &info ) )
            {
                return 0;
            }

            lua_newtable( L );

            lua_pushstring( L, "majorversion" );
            lua_pushnumber( L, (lua_Number)info.majorversion );
            lua_settable( L, -3 );

            lua_pushstring( L, "minorversion" );
            lua_pushnumber( L, (lua_Number)info.minorversion );
            lua_settable( L, -3 );

            lua_pushstring( L, "revision" );
            lua_pushnumber( L, (lua_Number)info.revision );
            lua_settable( L, -3 );

            lua_pushstring( L, "description" );
            lua_pushstring( L, info.description );
            lua_settable( L, -3 );

            if( info.isalloc )
            {
                free( (void *)info.description );
            }

            return 1;
        }

        /*************************************************************/

#ifdef _MSC_VER
#    pragma comment( lib, "version.lib" )
#endif

        int getKernelVersion( struct OsVersionInfo *info )
        {
            DWORD size = GetFileVersionInfoSizeA( "kernel32.dll", NULL );
            if( size > 0 )
            {
                void *data = malloc( size );
                if( GetFileVersionInfoA( "kernel32.dll", 0, size, data ) )
                {
                    void *fixedInfoPtr;
                    UINT  fixedInfoSize;
                    if( VerQueryValueA( data, "\\", &fixedInfoPtr, &fixedInfoSize ) )
                    {
                        VS_FIXEDFILEINFO *fileInfo = (VS_FIXEDFILEINFO *)fixedInfoPtr;
                        info->majorversion         = HIWORD( fileInfo->dwProductVersionMS );
                        info->minorversion         = LOWORD( fileInfo->dwProductVersionMS );
                        info->revision             = HIWORD( fileInfo->dwProductVersionLS );
                        return TRUE;
                    }
                }
            }
            return FALSE;
        }

        int getversion( struct OsVersionInfo *info )
        {
            HKEY key;
            info->description = "Windows";

            // First get a friendly product name from the registry.
            if( RegOpenKeyExA( HKEY_LOCAL_MACHINE, "Software\\Microsoft\\Windows NT\\CurrentVersion", 0, KEY_READ, &key ) ==
                ERROR_SUCCESS )
            {
                char  value[512];
                DWORD value_length = sizeof( value );
                DWORD type;
                RegQueryValueExA( key, "productName", NULL, &type, (LPBYTE)value, &value_length );
                RegCloseKey( key );
                if( type == REG_SZ )
                {
                    info->description = strdup( value );
                    info->isalloc     = 1;
                }
            }

            // See if we can get a product version number from kernel32.dll
            return getKernelVersion( info );
        }

        /*************************************************************/
        int os_is64bit( lua_State *L )
        {
            // If this code returns true, then the platform is 64-bit. If it
            // returns false, the platform might still be 64-bit, but more
            // checking will need to be done on the Lua side of things.
            typedef BOOL( WINAPI * WowFuncSig )( HANDLE, PBOOL );
            WowFuncSig func = (WowFuncSig)GetProcAddress( GetModuleHandle( TEXT( "kernel32" ) ), "IsWow64Process" );
            if( func )
            {
                BOOL isWow = FALSE;
                if( func( GetCurrentProcess(), &isWow ) )
                {
                    lua_pushboolean( L, isWow );
                    return 1;
                }
            }

            lua_pushboolean( L, 0 );
            return 1;
        }

        int do_locate( lua_State *L, const char *filename, const char *path )
        {
            if( do_pathsearch( L, filename, path ) )
            {
                lua_pushstring( L, "/" );
                lua_pushstring( L, filename );
                lua_concat( L, 3 );
                return 1;
            }

            return 0;
        }

        int os_locate( lua_State *L )
        {
            const char *path;
            int         i;
            int         nArgs = lua_gettop( L );

            /* Fetch premake.path */
            lua_getglobal( L, "premake" );
            lua_getfield( L, -1, "path" );
            path = lua_tostring( L, -1 );

            for( i = 1; i <= nArgs; ++i )
            {
                const char *name = lua_tostring( L, i );

                /* Direct path to an embedded file? */
                if( name[0] == '$' && name[1] == '/' && premake_find_embedded_script( name + 2 ) )
                {
                    lua_pushvalue( L, i );
                    return 1;
                }

                /* Direct path to file? Return as absolute path */
                if( do_isfile( L, name ) )
                {
                    lua_pushcfunction( L, path_getabsolute );
                    lua_pushvalue( L, i );
                    lua_call( L, 1, 1 );
                    return 1;
                }

                /* do_locate(arg[i], premake.path) */
                if( do_locate( L, name, path ) )
                {
                    return 1;
                }

                /* embedded in the executable? */
                if( premake_find_embedded_script( name ) )
                {
                    lua_pushstring( L, "$/" );
                    lua_pushvalue( L, i );
                    lua_concat( L, 2 );
                    return 1;
                }
            }

            return 0;
        }

        typedef struct struct_MatchInfo
        {
            HANDLE           handle;
            int              is_first;
            WIN32_FIND_DATAW entry;
        } MatchInfo;

        int os_matchstart( lua_State *L )
        {
            const char *mask = luaL_checkstring( L, 1 );
            MatchInfo  *m;

            wchar_t wide_mask[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, mask, -1, wide_mask, PATH_MAX ) == 0 )
            {
                lua_pushstring( L, "unable to encode mask" );
                return lua_error( L );
            }

            m = (MatchInfo *)malloc( sizeof( MatchInfo ) );

            m->handle   = FindFirstFileW( wide_mask, &m->entry );
            m->is_first = 1;
            lua_pushlightuserdata( L, m );
            return 1;
        }

        int os_matchdone( lua_State *L )
        {
            MatchInfo *m = (MatchInfo *)lua_touserdata( L, 1 );
            if( m->handle != INVALID_HANDLE_VALUE )
                FindClose( m->handle );
            free( m );
            return 0;
        }

        int os_matchname( lua_State *L )
        {
            MatchInfo *m = (MatchInfo *)lua_touserdata( L, 1 );

            char filename[PATH_MAX];
            if( WideCharToMultiByte( CP_UTF8, 0, m->entry.cFileName, -1, filename, PATH_MAX, NULL, NULL ) == 0 )
            {
                lua_pushstring( L, "unable to decode filename" );
                return lua_error( L );
            }

            lua_pushstring( L, filename );
            return 1;
        }

        int os_matchisfile( lua_State *L )
        {
            MatchInfo *m = (MatchInfo *)lua_touserdata( L, 1 );
            lua_pushboolean( L, ( m->entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY ) == 0 );
            return 1;
        }

        int os_matchnext( lua_State *L )
        {
            MatchInfo *m = (MatchInfo *)lua_touserdata( L, 1 );
            if( m->handle == INVALID_HANDLE_VALUE )
            {
                return 0;
            }

            while( m ) /* loop forever */
            {
                if( m->is_first )
                    m->is_first = 0;
                else
                {
                    if( !FindNextFileW( m->handle, &m->entry ) )
                        return 0;
                }

                if( wcscmp( m->entry.cFileName, L"." ) != 0 && wcscmp( m->entry.cFileName, L".." ) != 0 )
                {
                    lua_pushboolean( L, 1 );
                    return 1;
                }
            }

            return 0;
        }

        int do_pathsearch( lua_State *L, const char *filename, const char *path )
        {
            do
            {
                const char *split;

                /* look for the closest path separator ; or : */
                /* can't use : on windows because it breaks on C:\path */
                const char *semi = strchr( path, ';' );

                const char *full = NULL;

                if( !semi )
                {
                    split = full;
                }
                else if( !full )
                {
                    split = semi;
                }
                else
                {
                    split = ( semi < full ) ? semi : full;
                }

                /* push this piece of the full search string onto the stack */
                if( split )
                {
                    lua_pushlstring( L, path, split - path );
                }
                else
                {
                    lua_pushstring( L, path );
                }

                /* keep an extra copy around, so I can return it if I have a match */
                lua_pushvalue( L, -1 );

                /* append the filename to make the full test path */
                lua_pushstring( L, "/" );
                lua_pushstring( L, filename );
                lua_concat( L, 3 );

                /* test it - if it exists, return the absolute path */
                if( do_isfile( L, lua_tostring( L, -1 ) ) )
                {
                    lua_pop( L, 1 );
                    lua_pushcfunction( L, path_getabsolute );
                    lua_pushvalue( L, -2 );
                    lua_call( L, 1, 1 );
                    return 1;
                }

                /* no match, set up the next try */
                lua_pop( L, 2 );
                path = ( split ) ? split + 1 : NULL;
            } while( path );

            return 0;
        }

        int os_pathsearch( lua_State *L )
        {
            int i;

            const char *filename = luaL_checkstring( L, 1 );
            for( i = 2; i <= lua_gettop( L ); ++i )
            {
                if( lua_isnil( L, i ) )
                    continue;
                if( do_pathsearch( L, filename, luaL_checkstring( L, i ) ) )
                    return 1;
            }

            return 0;
        }

        int os_realpath( lua_State *L )
        {
            char result[PATH_MAX];
            int  ok;

            const char *path = luaL_checkstring( L, 1 );

            ok = ( _fullpath( result, path, PATH_MAX ) != NULL );
            do_getabsolute( result, path, NULL );
            ok = 1;

            if( !ok )
            {
                lua_pushnil( L );
                lua_pushfstring( L, "unable to fetch real path of '%s', errno %d : %s", path, errno, strerror( errno ) );
                return 2;
            }

            lua_pushstring( L, result );
            return 1;
        }

        static int compare_file( const char *content, size_t length, const char *dst )
        {
            FILE  *file;
            size_t size;
            size_t read;
            char   buffer[4096];
            size_t num;

            wchar_t wide_path[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, dst, -1, wide_path, PATH_MAX ) == 0 )
                return FALSE;

            file = _wfopen( wide_path, L"rb" );

            if( file == NULL )
            {
                return FALSE;
            }

            // check sizes.
            fseek( file, 0, SEEK_END );
            size = ftell( file );
            fseek( file, 0, SEEK_SET );

            if( length != size )
            {
                fclose( file );
                return FALSE;
            }

            while( size > 0 )
            {
                num = size > 4096 ? 4096 : size;

                read = fread( buffer, 1, num, file );
                if( read != num )
                {
                    fclose( file );
                    return FALSE;
                }

                if( memcmp( content, buffer, num ) != 0 )
                {
                    fclose( file );
                    return FALSE;
                }

                size -= num;
                content += num;
            }

            fclose( file );
            return TRUE;
        }

        int os_writefile_ifnotequal( lua_State *L )
        {
            FILE       *file;
            size_t      length;
            const char *content = luaL_checklstring( L, 1, &length );
            const char *dst     = luaL_checkstring( L, 2 );

            // if destination exist, and they are the same, no need to copy.
            if( do_isfile( L, dst ) && compare_file( content, length, dst ) )
            {
                lua_pushinteger( L, 0 );
                return 1;
            }

            wchar_t wide_path[PATH_MAX];
            if( MultiByteToWideChar( CP_UTF8, 0, dst, -1, wide_path, PATH_MAX ) == 0 )
                return FALSE;

            file = _wfopen( wide_path, L"wb" );

            if( file != NULL )
            {
                fwrite( content, 1, length, file );
                fclose( file );

                lua_pushinteger( L, 1 );
                return 1;
            }

            lua_pushinteger( L, -1 );
            lua_pushfstring( L, "unable to write file to '%s'", dst );
            return 2;
        }

    } // namespace details

    void open_os_library( sol::table &module )
    {
        // auto module = scriptingState["os"].get_or_create<sol::table>();
    }
} // namespace numlua::core