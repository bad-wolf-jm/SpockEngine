

#ifndef _FRAGMENT_SHADER_UNIFORM_INPUT_HPP_
#define _FRAGMENT_SHADER_UNIFORM_INPUT_HPP_
#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#    include "Common/HelperFunctions.hpp"
#    include "Common/LightData.hpp"
#    include "Common/ShaderMaterial.hpp"
#endif


// clang-format off
LAYOUT_UNIFORM_BUFFER( CAMERA_PARAMETERS_BIND_POINT, 0 ) __UNIFORM_BUFFER__ CameraParameters 
{ 
    float mExposure;
    float mGamma;
} gCamera;
// clang-format on

// Try to be as bindless as possible and bind all available textures and all materials
// in one go as an array.
// clang-format off
LAYOUT_UNIFORM_BUFFER( MATERIAL_DATA_BIND_POINT, 0 ) __UNIFORM_BUFFER__ ShaderMaterials 
{ 
    sShaderMaterial mArray[]; 
} gMaterials;
// clang-format on

LAYOUT_UNIFORM( MATERIAL_TEXTURES_BIND_POINT, 0 ) __UNIFORM__ sampler2D gTextures[];

#if !defined( SHADING_MODEL_UNLIT )
// clang-format off
LAYOUT_UNIFORM_BUFFER( DIRECTIONAL_LIGHTS_BIND_POINT, 0 ) __UNIFORM_BUFFER__ DirectionalLights
{
    sDirectionalLight mArray[];
} gDirectionalLights;

LAYOUT_UNIFORM_BUFFER( PUNCTUAL_LIGHTS_BIND_POINT, 0 ) __UNIFORM_BUFFER__ PointLights
{
    sPunctualLight mArray[];
} gPunctualLights;
// clang-format on
#endif

#if !defined( __cplusplus )
layout( push_constant ) uniform Material
{
    int mMaterialID;
}
gMaterialID;
#else
struct Material
{
    int mMaterialID;
} gMaterialID
#endif

#endif
