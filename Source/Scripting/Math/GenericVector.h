#pragma once

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include "Scripting/PrimitiveTypes.h"

namespace SE::Core
{
    using namespace sol;

    template <typename _VecType>
    void DeclareVectorOperation( table &aScriptState )
    {
        if constexpr( std::is_same<_VecType::value_type, float>::value )
        {
            aScriptState["normalize"] = []( _VecType aVector ) -> _VecType { return normalize( aVector ); };
            aScriptState["length"]    = []( _VecType aVector ) -> _VecType::value_type { return length( aVector ); };
            aScriptState["length2"]   = []( _VecType aVector ) -> _VecType::value_type { return length2( aVector ); };
            aScriptState["dist2"]     = []( _VecType aVector0, _VecType aVector1 ) -> _VecType::value_type
            { return dist2( aVector0, aVector1 ); };
            aScriptState["dot"] = []( _VecType aVector0, _VecType aVector1 ) -> _VecType::value_type
            { return dot( aVector0, aVector1 ); };
            aScriptState["mix"] = []( _VecType aVector0, _VecType aVector1, float aCoefficient ) -> _VecType
            { return mix<_VecType>( aVector0, aVector1, aCoefficient ); };
        }
    }

    template <typename _VecType>
    usertype<_VecType> NewVectorType( table &aScriptState, std::string aName )
    {
        auto lNewType = declare_primitive_type<_VecType>( aScriptState, aName );

        if constexpr( std::is_same<_VecType::value_type, float>::value )
        {
            lNewType["normalize"] = []( _VecType aSelf ) -> _VecType { return normalize( aSelf ); };
            lNewType["length"]    = []( _VecType aSelf ) -> _VecType::value_type { return length( aSelf ); };
            lNewType["length2"]   = []( _VecType aSelf ) -> _VecType::value_type { return length2( aSelf ); };
            lNewType["dist2"]     = []( _VecType aSelf, _VecType aOther ) -> _VecType::value_type { return dist2( aSelf, aOther ); };
            lNewType["dot"]       = []( _VecType aSelf, _VecType aOther ) -> _VecType::value_type { return dot( aSelf, aOther ); };
        }

        // clang-format off
        lNewType["__add"] = [&]( _VecType aSelf, _VecType aOther ) { return aSelf + aOther; };
        lNewType["__mul"] = overload(
            []( const _VecType &v1, const _VecType &v2 ) -> _VecType { return v1 * v2; },
            []( const _VecType &v1, _VecType::value_type f ) -> _VecType { return v1 * f; },
            []( _VecType::value_type f, const _VecType &v1 ) -> _VecType { return f * v1; } );
        // clang-format on

        return lNewType;
    }

} // namespace SE::Core