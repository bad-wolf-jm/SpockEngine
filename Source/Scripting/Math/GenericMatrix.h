#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>
#include "Scripting/PrimitiveTypes.h"

namespace LTSE::Core
{
    using namespace sol;

    template <typename _MatrixType> usertype<_MatrixType> NewMatrixType( table &aScriptState, std::string aName )
    {
        auto lNewType = DeclarePrimitiveType<_MatrixType>( aScriptState, aName );

        lNewType["inverse"]   = []( _MatrixType aSelf ) -> _MatrixType { return Inverse( aSelf ); };
        lNewType["transpose"] = []( _MatrixType aSelf ) -> _MatrixType { return Transpose( aSelf ); };

        //clang-format off
        lNewType["__add"] = []( const _MatrixType &v1, const _MatrixType &v2 ) -> _MatrixType { return v1 + v2; };
        lNewType["__mul"] = overload(
            []( const _MatrixType &v1, const _MatrixType &v2 ) -> _MatrixType { return v1 * v2; },
            []( const _MatrixType &v1, const _MatrixType::col_type &v2 ) -> _MatrixType::col_type { return v1 * v2; },
            []( const _MatrixType::row_type &v2, const _MatrixType &v1 ) -> _MatrixType::row_type { return v2 * v1; },
            []( const _MatrixType &v1, _MatrixType::value_type f ) -> _MatrixType { return v1 * f; },
            []( _MatrixType::value_type f, _MatrixType &v1 ) -> _MatrixType { return f * v1; }
        );
        //clang-format on
        return lNewType;
    }
}