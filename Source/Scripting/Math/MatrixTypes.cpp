#include "MatrixTypes.h"
#include "GenericMatrix.h"

#include "Core/Math/Types.h"

namespace SE::Core
{
    using namespace math;

    void define_matrix_types( sol::table &aModule )
    {
        auto lMat3Type = new_matrix_type<mat3>( aModule, "mat3" );
        // clang-format off
        lMat3Type[sol::call_constructor] =
            factories(
                []() { return mat3( 1.0f ); },
                []( const float &d ) { return mat3( d ); },
                []( const vec3 &d ) { return FromDiagonal( d ); },
                []( const vec3 &c1, const vec3 &c2, const vec3 &c3 ) { return mat3(c1, c2, c3); },
                []( const mat4 &c1 ) { return mat3(c1); }
            ) ;
        // clang-format on
        lMat3Type["comatrix"] = []( mat3 aSelf ) -> mat3 { return Comatrix( aSelf ); };

        auto lMat4Type = new_matrix_type<mat4>( aModule, "mat4" );

        // clang-format off
        lMat4Type[sol::call_constructor] =
            factories( []() { return mat4( 1.0f ); },
                []( const float &d ) { return mat4( d ); },
                []( const vec4 &d ) { return FromDiagonal( d ); },
                []( const vec4 &c1, const vec4 &c2, const vec4 &c3, const vec4 &c4 ) { return mat4(c1, c2, c3, c4); },
                []( mat3 const &aRotation, vec3 const &aTranslation ) { return FromComponents( aRotation, aTranslation ); }
        );
        // clang-format on
        lMat4Type["normal_matrix"]   = []( mat4 aSelf ) -> mat4 { return NormalMatrix( aSelf ); };
        lMat4Type["get_rotation"]    = []( mat4 aSelf ) -> mat3 { return Rotation( aSelf ); };
        lMat4Type["get_translation"] = []( mat4 aSelf ) -> vec3 { return Translation( aSelf ); };
        lMat4Type["get_scale"]       = []( mat4 aSelf ) -> vec3 { return Scaling( aSelf ); };
        lMat4Type["up"]              = []( mat4 aSelf ) -> vec3 { return UpDirection( aSelf ); };
        lMat4Type["right"]           = []( mat4 aSelf ) -> vec3 { return RightDirection( aSelf ); };
        lMat4Type["back"]            = []( mat4 aSelf ) -> vec3 { return BackwardDirection( aSelf ); };
    }
} // namespace SE::Core
