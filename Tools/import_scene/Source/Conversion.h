#pragma once

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "Core/Math/Types.h"

math::vec3 to_vec3( aiVector3D l_V );
math::vec3 to_vec3( aiColor3D l_V );
math::vec2 to_vec2( aiVector3D l_V );
math::vec4 to_vec4( aiColor4D l_V );
math::mat4 to_mat4( aiMatrix4x4 l_V );
math::quat to_quat( aiQuaternion l_V );
