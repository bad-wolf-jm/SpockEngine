#include "MatrixTypes.h"
#include "GenericMatrix.h"

#include "Core/Math/Types.h"

namespace SE::Core
{
    using namespace math;

    void define_matrix_types( sol::table &module )
    {
        auto mat3Type = new_matrix_type<mat3>( module, "mat3" );
        // clang-format off
        mat3Type[sol::call_constructor] =
            factories(
                []() { return mat3( 1.0f ); },
                []( const float &d ) { return mat3( d ); },
                []( const vec3 &d ) { return FromDiagonal( d ); },
                []( const vec3 &c1, const vec3 &c2, const vec3 &c3 ) { return mat3(c1, c2, c3); },
                []( const mat4 &c1 ) { return mat3(c1); }
            ) ;
        // clang-format on
        mat3Type["comatrix"] = []( mat3 self ) -> mat3 { return Comatrix( self ); };

        auto mat4Type = new_matrix_type<mat4>( module, "mat4" );

        // clang-format off
        mat4Type[sol::call_constructor] =
            factories( []() { return mat4( 1.0f ); },
                []( const float &d ) { return mat4( d ); },
                []( const vec4 &d ) { return FromDiagonal( d ); },
                []( const vec4 &c1, const vec4 &c2, const vec4 &c3, const vec4 &c4 ) { return mat4(c1, c2, c3, c4); },
                []( mat3 const &aRotation, vec3 const &aTranslation ) { return FromComponents( aRotation, aTranslation ); }
        );
        // clang-format on
        mat4Type["normal_matrix"]   = []( mat4 self ) -> mat4 { return NormalMatrix( self ); };
        mat4Type["get_rotation"]    = []( mat4 self ) -> mat3 { return Rotation( self ); };
        mat4Type["get_translation"] = []( mat4 self ) -> vec3 { return Translation( self ); };
        mat4Type["get_scale"]       = []( mat4 self ) -> vec3 { return Scaling( self ); };
        mat4Type["up"]              = []( mat4 self ) -> vec3 { return UpDirection( self ); };
        mat4Type["right"]           = []( mat4 self ) -> vec3 { return RightDirection( self ); };
        mat4Type["back"]            = []( mat4 self ) -> vec3 { return BackwardDirection( self ); };
    }
} // namespace SE::Core
