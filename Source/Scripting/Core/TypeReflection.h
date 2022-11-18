#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/EntityRegistry/Registry.h"

namespace SE::Core
{
    [[nodiscard]] entt::id_type GetTypeID( const sol::table &aObject );

    template <typename T> [[nodiscard]] entt::id_type DeduceType( T &&aObject )
    {
        switch( aObject.get_type() )
        {
        case sol::type::number:
            return aObject.as<entt::id_type>();
        case sol::type::table:
            return GetTypeID( aObject );
        }
        assert( false );
        return -1;
    }

    template <typename... Args> inline auto InvokeMetaFunction( entt::meta_type meta_type, entt::id_type function_id, Args &&...args )
    {
        if( !meta_type )
        {
            assert( false );
        }
        else
        {
            auto meta_function = meta_type.func( function_id );
            if( meta_function )
                return meta_function.invoke( {}, std::forward<Args>( args )... );
        }
        return entt::meta_any{};
    }

    template <typename... Args> inline auto InvokeMetaFunction( entt::id_type type_id, entt::id_type function_id, Args &&...args )
    {
        return InvokeMetaFunction( entt::resolve( type_id ), function_id, std::forward<Args>( args )... );
    }


}