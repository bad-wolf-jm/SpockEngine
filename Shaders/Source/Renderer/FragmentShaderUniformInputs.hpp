

#ifndef _FRAGMENT_SHADER_UNIFORM_INPUT_HPP_
#define _FRAGMENT_SHADER_UNIFORM_INPUT_HPP_
#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#    include "Common/HelperFunctions.hpp"
#    include "Common/LightInputData.hpp"
#    include "Common/ShaderMaterial.hpp"
#endif

// clang-format off
LAYOUT_UNIFORM( CAMERA_PARAMETERS_BIND_POINT, 0 ) CameraParameters 
{ 
    float mExposure;
    float mGamma;
    float3 mPosition;
    float mPadding;
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

LAYOUT_UNIFORM( MATERIAL_TEXTURES_BIND_POINT, 0 ) sampler2D gTextures[];

// clang-format off
LAYOUT_UNIFORM( DIRECTIONAL_LIGHTS_BIND_POINT, 0 ) DirectionalLight
{
    sDirectionalLight mData;
} gDirectionalLight;

LAYOUT_UNIFORM_BUFFER( PUNCTUAL_LIGHTS_BIND_POINT, 0 ) __UNIFORM_BUFFER__ PointLights
{
    sPunctualLight mArray[];
} gPunctualLights;
// clang-format on

LAYOUT_UNIFORM( DIRECTIONAL_LIGHTS_SHADOW_MAP_BIND_POINT, 0 ) sampler2D gDirectionalLightShadowMap;
LAYOUT_UNIFORM( PUNCTUAL_LIGHTS_SHADOW_MAP_BIND_POINT, 0 ) samplerCube gPunctualLightShadowMaps[];

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
