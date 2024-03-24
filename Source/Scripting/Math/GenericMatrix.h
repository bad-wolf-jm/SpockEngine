#pragma once
#define SOL_ALL_SAFETIES_ON 1
#include "Scripting/PrimitiveTypes.h"
#include <sol/sol.hpp>

namespace SE::Core
{
    using namespace sol;

    template <typename _MatrixType>
    usertype<_MatrixType> new_matrix_type( table &scriptState, std::string name )
    {
        auto lNewType = declare_primitive_type<_MatrixType>( scriptState, name );

        lNewType["inverse"]   = []( _MatrixType self ) -> _MatrixType { return Inverse( self ); };
        lNewType["transpose"] = []( _MatrixType self ) -> _MatrixType { return Transpose( self ); };

        //clang-format off
        lNewType["__add"] = []( const _MatrixType &v1, const _MatrixType &v2 ) -> _MatrixType { return v1 + v2; };
        lNewType["__mul"] =
            overload( []( const _MatrixType &v1, const _MatrixType &v2 ) -> _MatrixType { return v1 * v2; },
                      []( const _MatrixType &v1, const _MatrixType::col_type &v2 ) -> _MatrixType::col_type { return v1 * v2; },
                      []( const _MatrixType::row_type &v2, const _MatrixType &v1 ) -> _MatrixType::row_type { return v2 * v1; },
                      []( const _MatrixType &v1, _MatrixType::value_type f ) -> _MatrixType { return v1 * f; },
                      []( _MatrixType::value_type f, _MatrixType &v1 ) -> _MatrixType { return f * v1; } );
        //clang-format on
        return lNewType;
    }
} // namespace SE::Core