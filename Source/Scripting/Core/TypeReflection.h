#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Core/Entity/Collection.h"

namespace SE::Core
{
    [[nodiscard]] entt::id_type get_type_id( const sol::table &object );

    template <typename T>
    [[nodiscard]] entt::id_type deduce_type( T &&object )
    {
        switch( object.get_type() )
        {
        case sol::type::number:
            return object.as<entt::id_type>();
        case sol::type::table:
            return get_type_id( object );
        }
        assert( false );
        return -1;
    }

    template <typename... Args>
    inline auto invoke_meta_function( entt::meta_type meta_type, entt::id_type function_id, Args &&...args )
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

    template <typename... Args>
    inline auto invoke_meta_function( entt::id_type type_id, entt::id_type function_id, Args &&...args )
    {
        return invoke_meta_function( entt::resolve( type_id ), function_id, std::forward<Args>( args )... );
    }

} // namespace SE::Core