
#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>

#ifdef __cplusplus
extern "C"
{
#endif
#include "lauxlib.h"
#include "lprefix.h"
#include "lua.h"
#include "lualib.h"
#ifdef __cplusplus
}
#endif

#define l_likely( x )   luai_likely( x )
#define l_unlikely( x ) luai_unlikely( x )

namespace numlua::core::lua
{
    using state_t = lua_State;

    struct stack_index_t
    {
        int id = -1;
    };

    // clang-format off
    struct nil_t { };
    struct number_t { };
    using c_function_t = lua_CFunction;
    struct c_closure_t
    {
        c_function_t fn;
        int32_t nup;
    };

    struct function_t { };
    struct string_t { };
    struct integer_t { };
    struct light_user_data_t 
    {
        void* ptr = nullptr;
    };
    struct user_data_t 
    {
        void* ptr = nullptr;
    };

    struct table_t { };
    struct thread_t { };
    // clang-format on

    constexpr stack_index_t registry{ LUA_REGISTRYINDEX };
    constexpr stack_index_t up_value( int i )
    {
        return stack_index_t{ lua_upvalueindex( i ) };
    }

    void call( state_t *L, int nargs, int nresults );
    void pop( state_t *L, int n );
    void remove( state_t *L, stack_index_t const &n );
    void set_top( state_t *L, stack_index_t const &i );

    void        push( state_t *L, stack_index_t const &i );
    const char *push( state_t *L, const char *s );
    const char *push( state_t *L, const char *s, int len );
    void        push( state_t *L, bool s );
    void        push( state_t *L, int64_t s );
    void        push( state_t *L, uint8_t s );
    void        push( state_t *L, light_user_data_t const &s );
    void        push( state_t *L, c_function_t const &s );
    void        push( state_t *L, c_closure_t const &s );
    void        push( state_t *L, nil_t const &s );
    int         get_field( state_t *L, stack_index_t const &i, const char *fieldName );
    void        set_field( state_t *L, stack_index_t const &i, const char *fieldName );
    bool        get_or_create_table( state_t *L, stack_index_t const &i, const char *fieldName );
    void        raw_set( state_t *L, stack_index_t const &i, int64_t idx );
    int         raw_get( state_t *L, stack_index_t const &i, int64_t idx );
    const char *check_string( state_t *L, stack_index_t const &i );

    template <typename _Ty>
    _Ty cast_as( state_t *L, stack_index_t const &i )
    {
        if constexpr( std::is_same<_Ty, const char *>::value )
        {
            return lua_tostring( L, i.id );
        }

        if constexpr( std::is_same<_Ty, std::string_view>::value )
        {
            return std::string_view( lua_tostring( L, i.id ) );
        }

        if constexpr( std::is_same<_Ty, bool>::value )
        {
            return static_cast<bool>( lua_toboolean( L, i.id ) );
        }

        if constexpr( std::is_same<_Ty, double>::value )
        {
            return static_cast<double>( lua_tonumber( L, i.id ) );
        }

        if constexpr( std::is_same<_Ty, int64_t>::value )
        {
            return static_cast<double>( lua_tointeger( L, i.id ) );
        }

        if constexpr( std::is_same<_Ty, light_user_data_t>::value )
        {
            return light_user_data_t{ lua_touserdata( L, i.id ) };
        }
    }

    template <typename _Ty>
    _Ty cast_as( state_t *L, stack_index_t const &i, _Ty def )
    {
        if constexpr( std::is_same<_Ty, const char *>::value )
        {
            return luaL_optstring( L, i.id, def );
        }

        if constexpr( std::is_same<_Ty, std::string_view>::value )
        {
            return std::string_view( luaL_optstring( L, i.id, def.data() ) );
        }

        if constexpr( std::is_same<_Ty, std::string>::value )
        {
            return std::string( luaL_optstring( L, i.id, def.data() ) );
        }

        if constexpr( std::is_same<_Ty, bool>::value )
        {
            if(lua_isnoneornil(L, i.id))
                return def;

            return static_cast<bool>( lua_toboolean( L, i.id ) );
        }

        if constexpr( std::is_same<_Ty, double>::value )
        {
            return static_cast<double>( luaL_optnumber( L, i.id, def ) );
        }

        if constexpr( std::is_same<_Ty, int64_t>::value )
        {
            return static_cast<double>( luaL_optinteger( L, i.id, def ) );
        }
    }

    void rotate( state_t *L, stack_index_t const &i, int64_t n );

    template <typename _Ty>
    bool is( state_t *L, stack_index_t const &i )
    {
        if constexpr( std::is_same<_Ty, bool>::value )
        {
            return lua_isboolean( L, i.id );
        }

        if constexpr( std::is_same<_Ty, nil_t>::value )
        {
            return lua_isnil( L, i.id );
        }

        if constexpr( std::is_same<_Ty, number_t>::value )
        {
            return lua_isnumber( L, i.id );
        }

        if constexpr( std::is_same<_Ty, c_function_t>::value )
        {
            return lua_iscfunction( L, i.id );
        }

        if constexpr( std::is_same<_Ty, function_t>::value )
        {
            return lua_isfunction( L, i.id );
        }

        if constexpr( std::is_same<_Ty, string_t>::value )
        {
            return lua_isstring( L, i.id );
        }

        if constexpr( std::is_same<_Ty, integer_t>::value )
        {
            return lua_isinteger( L, i.id );
        }

        if constexpr( std::is_same<_Ty, light_user_data_t>::value )
        {
            return lua_islightuserdata( L, i.id );
        }

        if constexpr( std::is_same<_Ty, user_data_t>::value )
        {
            return lua_isuserdata( L, i.id );
        }

        if constexpr( std::is_same<_Ty, table_t>::value )
        {
            return lua_istable( L, i.id );
        }

        if constexpr( std::is_same<_Ty, table_t>::value )
        {
            return lua_isthread( L, i.id );
        }
    }

    template <typename... Args>
    int error( state_t *L, Args... s )
    {
        return luaL_error( L, std::forward<Args>( s )... );
    }

    void create_table( state_t *L, int narr, int nrecords );
    void create_table( state_t *L );
    void set_metatable( state_t *L, stack_index_t const &i );
} // namespace numlua::core::lua
