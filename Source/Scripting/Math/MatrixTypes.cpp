#include "MatrixTypes.h"
#include "GenericMatrix.h"

#include "Core/Math/Types.h"

namespace numlua::core
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
                []( const vec3 &d ) { return from_diagonal( d ); },
                []( const vec3 &c1, const vec3 &c2, const vec3 &c3 ) { return mat3(c1, c2, c3); },
                []( const mat4 &c1 ) { return mat3(c1); }
            ) ;
        // clang-format on
        mat3Type["comatrix"] = []( mat3 self ) -> mat3 { return comatrix( self ); };

        auto mat4Type = new_matrix_type<mat4>( module, "mat4" );

        // clang-format off
        mat4Type[sol::call_constructor] =
            factories( []() { return mat4( 1.0f ); },
                []( const float &d ) { return mat4( d ); },
                []( const vec4 &d ) { return from_diagonal( d ); },
                []( const vec4 &c1, const vec4 &c2, const vec4 &c3, const vec4 &c4 ) { return mat4(c1, c2, c3, c4); },
                []( mat3 const &aRotation, vec3 const &aTranslation ) { return from_components( aRotation, aTranslation ); }
        );
        // clang-format on
        mat4Type["normal_matrix"]   = []( mat4 self ) -> mat4 { return normal_matrix( self ); };
        mat4Type["get_rotation"]    = []( mat4 self ) -> mat3 { return rotation( self ); };
        mat4Type["get_translation"] = []( mat4 self ) -> vec3 { return translation( self ); };
        mat4Type["get_scale"]       = []( mat4 self ) -> vec3 { return scaling( self ); };
        mat4Type["up"]              = []( mat4 self ) -> vec3 { return up_direction( self ); };
        mat4Type["right"]           = []( mat4 self ) -> vec3 { return right_direction( self ); };
        mat4Type["back"]            = []( mat4 self ) -> vec3 { return backward_direction( self ); };
    }
} // namespace SE::Core
