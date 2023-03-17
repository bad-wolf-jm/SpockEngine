#pragma once

#include <type_traits>

#include "Core/EntityCollection/Collection.h"

namespace SE::Core
{
    template <typename... Args>
    inline auto InvokeMetaFunction( entt::meta_type aMetaType, entt::id_type aFunctionID, Args &&...aArgs )
    {
        if( !aMetaType )
        {
            assert( false );
        }
        else
        {
            auto lMetaFunction = aMetaType.func( aFunctionID );
            
            if( lMetaFunction ) return lMetaFunction.invoke( {}, std::forward<Args>( aArgs )... );
        }
        return entt::meta_any{};
    }

    template <typename... Args>
    inline auto InvokeMetaFunction( entt::id_type aTypeID, entt::id_type aFunctionID, Args &&...aArgs )
    {
        return InvokeMetaFunction( entt::resolve( aTypeID ), aFunctionID, std::forward<Args>( aArgs )... );
    }

}; // namespace SE::Core