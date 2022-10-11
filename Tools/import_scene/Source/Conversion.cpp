#include "Conversion.h"

math::vec3 to_vec3( aiVector3D l_V ) { return { l_V.x, l_V.y, l_V.z }; }
math::vec3 to_vec3( aiColor3D l_V ) { return { l_V.r, l_V.g, l_V.b }; }

math::vec2 to_vec2( aiVector3D l_V ) { return { l_V.x, l_V.y }; }

math::vec4 to_vec4( aiColor4D l_V ) { return { l_V.r, l_V.g, l_V.b, l_V.a }; }

math::quat to_quat( aiQuaternion l_V ) { return { l_V.w, l_V.x, l_V.y, l_V.z }; }

math::mat4 to_mat4( aiMatrix4x4 l_V )
{
    return math::Transpose(
        math::mat4{ { l_V.a1, l_V.a2, l_V.a3, l_V.a4 }, { l_V.b1, l_V.b2, l_V.b3, l_V.b4 }, { l_V.c1, l_V.c2, l_V.c3, l_V.c4 }, { l_V.d1, l_V.d2, l_V.d3, l_V.d4 } } );
}
