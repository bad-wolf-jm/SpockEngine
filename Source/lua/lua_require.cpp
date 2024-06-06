#pragma once

#include <fmt/format.h>
#include <iostream>
#include <ostream>
#define loadlib_c
#define LUA_LIB

#include "lua_api.h"
#include "lua/libs/strings.h"
#include "lua/libs/path.h"
#include "lua/libs/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

/*
** LUA_IGMARK is a mark to ignore all before it when building the
** luaopen_ function name.
*/
#if !defined( LUA_IGMARK )
#    define LUA_IGMARK "-"
#endif

#if !defined( LUA_LLE_FLAGS )
#    define LUA_LLE_FLAGS 0
#endif

/*
** LUA_CSUBSEP is the character that replaces dots in submodule names
** when searching for a C loader.
** LUA_LSUBSEP is the character that replaces dots in submodule names
** when searching for a Lua loader.
*/
#if !defined( LUA_CSUBSEP )
#    define LUA_CSUBSEP LUA_DIRSEP
#endif

#if !defined( LUA_LSUBSEP )
#    define LUA_LSUBSEP LUA_DIRSEP
#endif

/* prefix for open functions in C libraries */
#define LUA_POF "luaopen_"

/* separator for open functions in C libraries */
#define LUA_OFSEP "_"

/*
** key for table in the registry that keeps handles
** for all loaded C libraries
*/
const char *const CLIBS = "_CLIBS";

#define LIB_FAIL "open"

namespace numlua::core::lib
{
    struct dll_handle_t
    {
        void *ptr;
        dll_handle_t( const char *path );
        ~dll_handle_t();
        lua::c_function_t get_symbol( const char *name );

        int         error_code = 0;
        std::string error_string;

        bool is_valid()
        {
            return ptr != nullptr;
        }

        void set_error()
        {
            error_code = GetLastError();
            char buffer[128];
            if( FormatMessageA( FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, NULL, error_code, 0, buffer,
                                sizeof( buffer ) / sizeof( char ), NULL ) )
            {
                error_string = std::string( buffer );
            }
            else
            {
                error_string = fmt::format( "system error {}\n", error_code );
            }
        }
    };

    dll_handle_t::~dll_handle_t()
    {
        if( ptr != nullptr )
            FreeLibrary( (HMODULE)ptr );
    }

    dll_handle_t ::dll_handle_t( const char *path )
    {
        HMODULE lib = LoadLibraryExA( path, NULL, LUA_LLE_FLAGS );

        ptr = lib;
    }

    lua::c_function_t dll_handle_t::get_symbol( const char *name )
    {
        if( ptr == nullptr )
            return nullptr;

        lua::c_function_t f = (lua::c_function_t)GetProcAddress( (HMODULE)ptr, name );
        return f;
    }
    /*
    ** Special type equivalent to '(void*)' for functions in gcc
    ** (to suppress warnings when converting function pointers)
    */
    typedef void ( *voidf )( void );

    /*
    ** Replace in the path (on the top of the stack) any occurrence
    ** of LUA_EXEC_DIR with the executable's path.
    */
    void setprogdir( lua::state_t *L )
    {
        char  buff[MAX_PATH + 1];
        char *lb;
        DWORD nsize = sizeof( buff ) / sizeof( char );
        DWORD n     = GetModuleFileNameA( NULL, buff, nsize ); /* get exec. name */
        if( n == 0 || n == nsize || ( lb = strrchr( buff, '\\' ) ) == NULL )
            lua::error( L, "unable to get ModuleFileName" );
        else
        {
            *lb = '\0'; /* cut name on the last '\\' to get the path */
            luaL_gsub( L, lua::cast_as<const char *>( L, lua::stack_index_t{ -1 } ), LUA_EXEC_DIR, buff );
            lua::remove( L, lua::stack_index_t{ -2 } ); /* remove original string */
        }
    }

    void pusherror( lua::state_t *L )
    {
        int  error = GetLastError();
        char buffer[128];
        if( FormatMessageA( FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, NULL, error, 0, buffer,
                            sizeof( buffer ) / sizeof( char ), NULL ) )
            lua::push( L, buffer );
        else
        {
            auto const s = fmt::format( "system error {}\n", error );
            lua::push( L, s.c_str() );
        }
    }

    void set_path( lua::state_t *L, const char *fieldname, const char *dft )
    {
        lua::push( L, dft ); /* use default */

        setprogdir( L );
        lua::set_field( L, lua::stack_index_t{ -2 }, fieldname ); /* package[fieldname] = path value */
    }

    /*
    ** return registry.CLIBS[path]
    */
    dll_handle_t *checkclib( lua::state_t *L, const char *path )
    {
        // void *plib;
        lua::get_field( L, lua::registry, CLIBS );
        lua::get_field( L, lua::stack_index_t{ -1 }, path );
        auto const &plib = lua::cast_as<lua::light_user_data_t>( L, lua::stack_index_t{ -1 } ); /* plib = CLIBS[path] */
        lua::pop( L, 2 );                                                                       /* pop CLIBS table and 'plib' */

        return static_cast<dll_handle_t *>( plib.ptr );
    }

    /*
    ** registry.CLIBS[path] = plib        -- for queries
    ** registry.CLIBS[#CLIBS + 1] = plib  -- also keep a list of all libraries
    */
    void addtoclib( lua::state_t *L, const char *path, dll_handle_t *plib )
    {
        lua::get_field( L, lua::registry, CLIBS );
        lua::push( L, lua::light_user_data_t{ plib } );
        lua::push( L, lua::stack_index_t{ -1 } );
        lua::set_field( L, lua::stack_index_t{ -3 }, path );                /* CLIBS[path] = plib */
        lua::raw_set( L, lua::stack_index_t{ -2 }, luaL_len( L, -2 ) + 1 ); /* CLIBS[#CLIBS + 1] = plib */
        lua::pop( L, 1 );                                                   /* pop CLIBS table */
    }

/* error codes for 'find_function_name' */
#define ERRLIB  1
#define ERRFUNC 2
    int readable( const char *filename )
    {
        FILE *f = fopen( filename, "r" ); /* try to open file */
        if( f == NULL )
            return 0; /* open failed */
        fclose( f );

        return 1;
    }

    /*
    ** Get the next name in '*path' = 'name1;name2;name3;...', changing
    ** the ending ';' to '\0' to create a zero-terminated string. Return
    ** NULL when list ends.
    */
    const char *getnextfilename( char **path, char *end )
    {
        char *sep;
        char *name = *path;
        if( name == end )
            return NULL; /* no more names */
        else if( *name == '\0' )
        {                          /* from previous iteration? */
            *name = *LUA_PATH_SEP; /* restore separator */
            name++;                /* skip it */
        }
        sep = strchr( name, *LUA_PATH_SEP ); /* find next separator */
        if( sep == NULL )                    /* separator not found? */
            sep = end;                       /* name goes until the end */
        *sep  = '\0';                        /* finish file name */
        *path = sep;                         /* will start next search from here */
        return name;
    }

    /*
    ** Given a path such as ";blabla.so;blublu.so", pushes the string
    **
    ** no file 'blabla.so'
    **	no file 'blublu.so'
    */
    void pusherrornotfound( lua::state_t *L, const char *path )
    {
        luaL_Buffer b;
        luaL_buffinit( L, &b );
        luaL_addstring( &b, "no file '" );
        luaL_addgsub( &b, path, LUA_PATH_SEP, "'\n\tno file '" );
        luaL_addstring( &b, "'" );
        luaL_pushresult( &b );
    }

    const char *searchpath( lua::state_t *L, const char *name, const char *path, const char *sep, const char *dirsep )
    {
        luaL_Buffer buff;
        char       *pathname;    /* path with name inserted */
        char       *endpathname; /* its end */
        const char *filename;
        /* separator is non-empty and appears in 'name'? */
        if( *sep != '\0' && strchr( name, *sep ) != NULL )
            name = luaL_gsub( L, name, sep, dirsep ); /* replace it by 'dirsep' */
        luaL_buffinit( L, &buff );
        /* add path to the buffer, replacing marks ('?') with the file name */
        luaL_addgsub( &buff, path, LUA_PATH_MARK, name );
        luaL_addchar( &buff, '\0' );
        pathname    = luaL_buffaddr( &buff ); /* writable list of file names */
        endpathname = pathname + luaL_bufflen( &buff ) - 1;
        while( ( filename = getnextfilename( &pathname, endpathname ) ) != NULL )
        {
            if( readable( filename ) )           /* does file exist and is readable? */
                return lua::push( L, filename ); /* save and return name */
        }
        luaL_pushresult( &buff );                                                          /* push path to create error message */
        pusherrornotfound( L, lua::cast_as<const char *>( L, lua::stack_index_t{ -1 } ) ); /* create error message */
        return NULL;                                                                       /* not found */
    }

    const char *findfile( lua::state_t *L, const char *name, const char *pname, const char *dirsep )
    {
        const char *path;
        lua::get_field( L, lua::up_value( 1 ), pname );
        path = lua::cast_as<const char *>( L, lua::stack_index_t{ -1 } );

        if( l_unlikely( path == NULL ) )
            lua::error( L, "'package.%s' must be a string", pname );

        return searchpath( L, name, path, ".", dirsep );
    }

    int checkload( lua::state_t *L, int stat, const char *filename )
    {
        if( l_likely( stat ) )
        {                             /* module loaded successfully? */
            lua::push( L, filename ); /* will be 2nd argument to module */

            return 2; /* return open function and file name */
        }
        else
        {
            return lua::error( L, "error loading module '%s' from file '%s':\n\t%s",
                               lua::cast_as<const char *>( L, lua::stack_index_t{ 1 } ), filename,
                               lua::cast_as<const char *>( L, lua::stack_index_t{ -1 } ) );
        }
    }

    int searcher_preload( lua::state_t *L )
    {
        const char *name = lua::check_string( L, lua::stack_index_t{ 1 } );
        lua::get_field( L, lua::registry, LUA_PRELOAD_TABLE );
        if( lua::get_field( L, lua::stack_index_t{ -1 }, name ) == LUA_TNIL )
        {
            /* not found? */
            auto const &s = fmt::format( "no field package.preload['{}']", name );
            lua::push( L, s.c_str() );

            return 1;
        }
        else
        {
            lua::push( L, ":preload:" );

            return 2;
        }
    }

    int searcher_Lua( lua::state_t *L )
    {
        const char *filename;
        const char *name = lua::check_string( L, lua::stack_index_t{ 1 } );
        filename         = findfile( L, name, "path", LUA_LSUBSEP );
        if( filename == NULL )
            return 1; /* module not found in this path */

        return checkload( L, ( luaL_loadfile( L, filename ) == LUA_OK ), filename );
    }

    /*
    ** Try to find a load function for module 'modname' at file 'filename'.
    ** First, change '.' to '_' in 'modname'; then, if 'modname' has
    ** the form X-Y (that is, it has an "ignore mark"), build a function
    ** name "luaopen_X" and look for it. (For compatibility, if that
    ** fails, it also tries "luaopen_Y".) If there is no ignore mark,
    ** look for a function named "luaopen_modname".
    */
    std::string find_module_load_function( lua::state_t *L, const char *modname )
    {
        const char *openfunc;
        const char *mark;
        modname = luaL_gsub( L, modname, ".", LUA_OFSEP );
        mark    = strchr( modname, *LUA_IGMARK );
        if( mark )
        {
            std::string prefix( modname, mark - modname );
            return fmt::format( LUA_POF "{}", openfunc );
        }

        return fmt::format( LUA_POF "{}", modname );
    }

    int searcher_C( lua::state_t *L )
    {
        const char *name     = lua::check_string( L, lua::stack_index_t{ 1 } );
        const char *filename = findfile( L, name, "cpath", LUA_CSUBSEP );
        if( filename == NULL )
            return 1; /* module not found in this path */

        dll_handle_t *lib = checkclib( L, filename );
        if( lib == nullptr )
        {
            lib = new dll_handle_t( filename );
            if( !lib->is_valid() )
            {
                // This should not happen
                delete lib;
                return 0;
            }
        }

        auto              module_load_name = find_module_load_function( L, name );
        lua::c_function_t func             = lib->get_symbol( module_load_name.c_str() );

        if( func == nullptr )
        {
            delete lib;
            return 0;
        }

        // Add module to _CLIBS and return the function
        addtoclib( L, filename, lib );
        lua::push( L, func );
        lua::push( L, filename );

        return 2;
    }

    static void find_loader( lua::state_t *L, const char *name )
    {
        int         i;
        luaL_Buffer msg; /* to build error message */
        /* push 'package.searchers' to index 3 in the stack */
        if( l_unlikely( lua::get_field( L, lua::up_value( 1 ), "searchers" ) != LUA_TTABLE ) )
            lua::error( L, "'package.searchers' must be a table" );

        luaL_buffinit( L, &msg );
        /*  iterate over available searchers to find a loader */
        for( i = 1;; i++ )
        {
            luaL_addstring( &msg, "\n\t" ); /* error-message prefix */
            if( l_unlikely( lua::raw_get( L, lua::stack_index_t{ 3 }, i ) == LUA_TNIL ) )
            {                            /* no more searchers? */
                lua::pop( L, 1 );        /* remove nil */
                luaL_buffsub( &msg, 2 ); /* remove prefix */
                luaL_pushresult( &msg ); /* create error message */
                lua::error( L, "module '%s' not found:%s", name, lua::cast_as<const char *>( L, lua::stack_index_t{ -1 } ) );
            }

            lua::push( L, name );
            lua::call( L, 1, 2 );                                         /* call it */
            if( lua::is<lua::function_t>( L, lua::stack_index_t{ -2 } ) ) /* did it find a loader? */
            {
                return; /* module loader found */
            }
            else if( lua::is<lua::string_t>( L, lua::stack_index_t{ -2 } ) )
            {                          /* searcher returned error message? */
                lua::pop( L, 1 );      /* remove extra return */
                luaL_addvalue( &msg ); /* concatenate error message */
            }
            else
            {                            /* no error message */
                lua::pop( L, 2 );        /* remove both returns */
                luaL_buffsub( &msg, 2 ); /* remove prefix */
            }
        }
    }

    static int ll_require( lua::state_t *L )
    {
        const char *name = lua::check_string( L, lua::stack_index_t{ 1 } );
        lua::set_top( L, lua::stack_index_t{ 1 } ); /* LOADED table will be at index 2 */
        lua::get_field( L, lua::registry, LUA_LOADED_TABLE );
        lua::get_field( L, lua::stack_index_t{ 2 }, name );     /* LOADED[name] */
        if( lua::cast_as<bool>( L, lua::stack_index_t{ -1 } ) ) /* is it there? */
            return 1;                                           /* package is already loaded */

        /* else must load package */
        lua::pop( L, 1 ); /* remove 'getfield' result */
        find_loader( L, name );
        lua::rotate( L, lua::stack_index_t{ -2 }, 1 ); /* function <-> loader data */
        lua::push( L, lua::stack_index_t{ 1 } );       /* name is 1st argument to module loader */
        lua::push( L, lua::stack_index_t{ -3 } );      /* loader data is 2nd argument */
        /* stack: ...; loader data; loader function; mod. name; loader data */
        lua::call( L, 2, 1 ); /* run loader to load module */

        /* stack: ...; loader data; result from loader */
        if( !lua::is<lua::nil_t>( L, lua::stack_index_t{ -1 } ) ) /* non-nil return? */
            lua::set_field( L, lua::stack_index_t{ 2 }, name );   /* LOADED[name] = returned value */
        else
            lua::pop( L, 1 ); /* pop nil */

        if( lua::get_field( L, lua::stack_index_t{ 2 }, name ) == LUA_TNIL )
        {
            /* module set no value? */
            lua::push( L, true );                               /* use true as result */
            lua_copy( L, -1, -2 );                              /* replace loader result */
            lua::set_field( L, lua::stack_index_t{ 2 }, name ); /* LOADED[name] = true */
        }

        lua::rotate( L, lua::stack_index_t{ -2 }, 1 ); /* loader data <-> module result  */

        return 2; /* return module result and loader data */
    }

    /* }====================================================== */

    const luaL_Reg pk_funcs[] = {
        // { "loadlib", ll_loadlib },
        // { "searchpath", ll_searchpath },
        /* placeholders */
        { "preload", NULL },   //
        { "cpath", NULL },     //
        { "path", NULL },      //
        { "searchers", NULL }, //
        { "loaded", NULL },    //
        { NULL, NULL } };

    const luaL_Reg ll_funcs[] = { { "require", ll_require }, { NULL, NULL } };

    static void create_searchers_table( lua::state_t *L )
    {
        const lua::c_function_t searchers[] = { searcher_preload, searcher_Lua, searcher_C, NULL };

        /* create 'searchers' table */
        lua::create_table( L, sizeof( searchers ) / sizeof( searchers[0] ) - 1, 0 );

        /* fill it with predefined searchers */
        for( int i = 0; searchers[i] != NULL; i++ )
        {
            lua::push( L, lua::stack_index_t{ -2 } ); /* set 'package' as upvalue for all searchers */
            lua::push( L, lua::c_closure_t{ searchers[i], 1 } );
            lua::raw_set( L, lua::stack_index_t{ -2 }, i + 1 );
        }

        lua::set_field( L, lua::stack_index_t{ -2 }, "searchers" ); /* put it in field 'searchers' */
    }

    /*
    ** __gc tag method for CLIBS table: calls 'lsys_unloadlib' for all lib
    ** handles in list CLIBS
    */
    static int gctm( lua::state_t *L )
    {
        lua_Integer n = luaL_len( L, 1 );
        for( ; n >= 1; n-- )
        {                                                  /* for each handle, in reverse order */
            lua::raw_get( L, lua::stack_index_t{ 1 }, n ); /* get handle CLIBS[n] */
            delete(dll_handle_t *)lua::cast_as<lua::light_user_data_t>( L, lua::stack_index_t{ -1 } ).ptr;
            lua::pop( L, 1 ); /* pop handle */
        }

        return 0;
    }

    /*
    ** create table CLIBS to keep track of loaded C libraries,
    ** setting a finalizer to close all libraries when closing state.
    */
    static void create_clibs_table( lua::state_t *L )
    {
        lua::get_or_create_table( L, lua::registry, CLIBS ); /* create CLIBS table */
        lua::create_table( L, 0, 1 );                        /* create metatable for CLIBS */
        lua::push( L, gctm );
        lua::set_field( L, lua::stack_index_t{ -2 }, "__gc" ); /* set finalizer for CLIBS table */
        lua::set_metatable( L, lua::stack_index_t{ -2 } );
    }

    int luaopen_package( lua::state_t *L, const char *path, const char *cpath )
    {
        // Create a table that will maintain a list of opened DLL's
        // so they can be closed when the state gets deleted.
        create_clibs_table( L );

        luaL_newlib( L, pk_funcs ); /* create 'package' table */

        // Create a table containing a list of functions which will search the
        // various paths. Each function return
        create_searchers_table( L );

        /* set paths */
        set_path( L, "path", path );
        set_path( L, "cpath", cpath );
        // /* store config information */

        /* set field 'loaded' */
        lua::get_or_create_table( L, lua::registry, LUA_LOADED_TABLE );
        lua::set_field( L, lua::stack_index_t{ -2 }, "loaded" );

        lua::get_or_create_table( L, lua::registry, LUA_PRELOAD_TABLE );
        // Add the standard libraries to the preload table
        const std::tuple<const char *, lua::c_function_t> builtinLibraries[] = {
            { LUA_COLIBNAME, luaopen_coroutine },    { LUA_TABLIBNAME, luaopen_table },       { LUA_IOLIBNAME, luaopen_io },
            { LUA_OSLIBNAME, luaopen_os },           { LUA_STRLIBNAME, lib::luaopen_string }, { LUA_MATHLIBNAME, luaopen_math },
            { LUA_UTF8LIBNAME, luaopen_utf8 },       { LUA_DBLIBNAME, luaopen_debug },        { "path", lib::open_path_library },
            { "logging", lib::open_logging_library } };

        for( auto const &[name, func] : builtinLibraries )
        {
            lua::push( L, func );
            lua::set_field( L, lua::stack_index_t{ -2 }, name );
        }
        lua::pop( L, 1 );
        // Add the 'require' function to the global namespace
        lua_pushglobaltable( L );
        lua::push( L, lua::stack_index_t{ -2 } ); /* set 'package' as upvalue for next lib */
        luaL_setfuncs( L, ll_funcs, 1 );          /* open lib into global table */
        lua::pop( L, 1 );                         /* pop global table */

        return 1; /* return 'package' table */
    }
} // namespace numlua::core::lib
