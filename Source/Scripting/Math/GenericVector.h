#pragma once

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Scripting/PrimitiveTypes.h"

namespace SE::Core
{
    using namespace sol;

    template <typename _VecType>
    void declare_vector_operation( table &scriptState )
    {
        if constexpr( std::is_same<_VecType::value_type, float>::value )
        {
            scriptState["normalize"] = []( _VecType vector ) -> _VecType { return normalize( vector ); };
            scriptState["length"]    = []( _VecType vector ) -> _VecType::value_type { return length( vector ); };
            scriptState["length2"]   = []( _VecType vector ) -> _VecType::value_type { return length2( vector ); };
            scriptState["dist2"]     = []( _VecType vector0, _VecType vector1 ) -> _VecType::value_type
            { return dist2( vector0, vector1 ); };
            scriptState["dot"] = []( _VecType vector0, _VecType vector1 ) -> _VecType::value_type
            { return dot( vector0, vector1 ); };
            scriptState["mix"] = []( _VecType vector0, _VecType vector1, float coefficient ) -> _VecType
            { return mix<_VecType>( vector0, vector1, coefficient ); };
        }
    }

    template <typename _VecType>
    usertype<_VecType> new_vector_type( table &scriptState, std::string name )
    {
        auto newType = declare_primitive_type<_VecType>( scriptState, name );

        if constexpr( std::is_same<_VecType::value_type, float>::value )
        {
            newType["normalize"] = []( _VecType self ) -> _VecType { return normalize( self ); };
            newType["length"]    = []( _VecType self ) -> _VecType::value_type { return length( self ); };
            newType["length2"]   = []( _VecType self ) -> _VecType::value_type { return length2( self ); };
            newType["dist2"]     = []( _VecType self, _VecType other ) -> _VecType::value_type { return dist2( self, other ); };
            newType["dot"]       = []( _VecType self, _VecType other ) -> _VecType::value_type { return dot( self, other ); };
        }

        // clang-format off
        newType["__add"] = [&]( _VecType self, _VecType other ) { return self + other; };
        newType["__mul"] = overload(
            []( const _VecType &v1, const _VecType &v2 ) -> _VecType { return v1 * v2; },
            []( const _VecType &v1, _VecType::value_type f ) -> _VecType { return v1 * f; },
            []( _VecType::value_type f, const _VecType &v1 ) -> _VecType { return f * v1; } );
        // clang-format on

        return newType;
    }

} // namespace SE::Core