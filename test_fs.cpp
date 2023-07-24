#version 450


#extension GL_EXT_nonuniform_qualifier : enable
#define __GLSL__
#define VULKAN_SEMANTICS
#define SHADING_MODEL_STANDARD
#define MATERIAL_HAS_UV0
#define MATERIAL_HAS_NORMALS
#define MATERIAL_HAS_BASE_COLOR_TEXTURE
#ifndef _DEFINITIONS_H_
#define _DEFINITIONS_H_

#if defined( VULKAN_SEMANTICS )
#    define LAYOUT_LOCATION( x )          layout( location = x )
#    define LAYOUT_UNIFORM( s, b )        layout( set = s, binding = b )
#    define LAYOUT_UNIFORM_BUFFER( s, b ) layout( set = s, binding = b ) readonly buffer
#    define __UNIFORM__                   uniform
#    define __UNIFORM_BUFFER__
#else
#    define LAYOUT_LOCATION( x )
#    define LAYOUT_UNIFORM( s, b )
#    define LAYOUT_UNIFORM_BUFFER( s, b )
#    define __UNIFORM__
#    define __UNIFORM_BUFFER__ struct
#endif

#if defined( __GLSL__ ) || defined( __cplusplus )
#    define float2   vec2
#    define float3   vec3
#    define float4   vec4
#    define float3x3 mat3
#    define float4x4 mat4
#    if defined( __cplusplus )
typedef struct _sampler2D sampler2D;
#    endif
#endif

#if defined( __cplusplus )
#    include "Core/Math/Types.h"
inline float4 gl_Position;
#endif

#define VIEW_PARAMETERS_BIND_POINT    0
#define CAMERA_PARAMETERS_BIND_POINT  1
#define MATERIAL_DATA_BIND_POINT      2
#define MATERIAL_TEXTURES_BIND_POINT  3
#define DIRECTIONAL_LIGHTS_BIND_POINT 4
#define PUNCTUAL_LIGHTS_BIND_POINT    5

#endif

#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#    define __SHADER_INPUT__
#    define __SHADER_OUTPUT__
#else
#    define __SHADER_INPUT__  in
#    define __SHADER_OUTPUT__ out
#endif

LAYOUT_LOCATION( 0 ) __SHADER_INPUT__ float3 inPos;

#if defined( MATERIAL_HAS_NORMALS )
LAYOUT_LOCATION( 1 ) __SHADER_INPUT__ float3 inNormal;
#endif

// Clever way of packing two UV channels into a single shader input
#if defined( MATERIAL_HAS_UV0 ) && !defined( MATERIAL_HAS_UV1 )
LAYOUT_LOCATION( 2 ) __SHADER_INPUT__ float2 inUV;
#elif defined( MATERIAL_HAS_UV1 )
LAYOUT_LOCATION( 2 ) __SHADER_INPUT__ float4 inUV;
#endif

#ifndef _LIGHT_DATA_HPP_
#define _LIGHT_DATA_HPP_

#if defined( __cplusplus )
#    include "Definitions.hpp"
#endif

// Shared with engine renderer code

struct sDirectionalLight
{
    float4 mBaseColorFactor;
};

struct sPunctualLight
{
    float4 mBaseColorFactor;
};

#endif
#ifndef _SHADER_MATERIAL_H_
#define _SHADER_MATERIAL_H_

#if defined( __cplusplus )
#    include "Definitions.hpp"
#endif

// Shared with engine renderer code

struct sShaderMaterial
{
    float4 mBaseColorFactor;
    float  mMetallicFactor;
    float  mRoughnessFactor;
    float  mOcclusionStrength;
    float3 mEmissiveFactor;
    int    mBaseColorUVChannel;
    int    mBaseColorTextureID;
    int    mEmissiveUVChannel;
    int    mEmissiveTextureID;
    int    mNormalUVChannel;
    int    mNormalTextureID;
    int    mMetalnessUVChannel;
    int    mMetalnessTextureID;
    int    mOcclusionUVChannel;
    int    mOcclusionTextureID;
};

#endif


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

// Try to be as bindless as possible and bind all available textures andd all materials
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

#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#endif

float4 SRGBtoLINEAR( float4 srgbIn )
{
    float3 bLess = step( float3( 0.04045 ), srgbIn.xyz );
    float3 linOut =
        mix( srgbIn.xyz / float3( 12.92 ), pow( ( srgbIn.xyz + float3( 0.055 ) ) / float3( 1.055 ), float3( 2.4 ) ), bLess );

    return float4( linOut, srgbIn.w );
}

#if defined( MATERIAL_HAS_UV0 )
float3 GetNormalFromMap( int aTexID, int aUVChannel )
{
    // // Perturb normal, see http://www.thetenthplanet.de/archives/1180
#    if defined( MATERIAL_HAS_UV0 ) && !defined( MATERIAL_HAS_UV1 )
    float3 tangentNormal = texture( gTextures[aTexID], inUV ).xyz * 2.0 - float3( 1.0 );
#    else
    float3 tangentNormal = texture( gTextures[aTexID], ( aUVChannel == 0 ) ? inUV.xy : inUV.zw ).xyz * 2.0 - float3( 1.0 );
#    endif
    float3   q1  = dFdx( inPos );
    float3   q2  = dFdy( inPos );
    float2   st1 = dFdx( inUV.xy );
    float2   st2 = dFdy( inUV.xy );
    float3   N   = normalize( inNormal );
    float3   T   = normalize( q1 * st2.t - q2 * st1.t );
    float3   B   = -normalize( cross( N, T ) );
    float3x3 TBN = float3x3( T, B, N );

    return normalize( TBN * tangentNormal );
}
#endif




#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#    include "Common/HelperFunctions.hpp"
#endif

struct MaterialInputs
{
    float4 mBaseColor;
#if defined( MATERIAL_HAS_NORMALS )
    float3 mNormal;
#endif

#if !defined( SHADING_MODEL_UNLIT )
#    if !defined( SHADING_MODEL_UNLIT )
    float mRoughness;
#    endif
#    if !defined( SHADING_MODEL_CLOTH )
    float mMetallic;
    float mReflectance;
#    endif
    float mAmbientOcclusion;
#endif

#if defined( MATERIAL_IS_EMISSIVE )
    float4 mEmissive;
#endif

#if !defined( SHADING_MODEL_CLOTH ) && !defined( SHADING_MODEL_SUBSURFACE ) && !defined( SHADING_MODEL_UNLIT )
    float3 mSheenColor;
    float  mSheenRoughness;
#endif

#if defined( MATERIAL_HAS_CLEARCOAT )
    float mClearCoat;
    float mClearCoatRoughness;
#endif

#if defined( MATERIAL_HAS_ANIROTROPY )
    float  mAnisotropy;
    float3 mAnisotropyDirection;
#endif

#if defined( SHADING_MODEL_SUBSURFACE ) || defined( MATERIAL_HAS_REFRACTION )
    float mThickness;
#endif

#if defined( SHADING_MODEL_SUBSURFACE )
    float3 mSubsurfaceColor;
    float  mSubsurfacePower;
#endif

#if defined( SHADING_MODEL_CLOTH )
    float3 mSheenColor;
#    if defined( MATERIAL_HAS_SUBSURFACE_COLOR )
    float3 mSubsurfaceColor;
#    endif
#endif

#if defined( MATERIAL_HAS_BENT_NORMAL )
    float3 mBentNormal;
#endif

#if defined( MATERIAL_HAS_CLEAR_COAT ) && defined( MATERIAL_HAS_CLEAR_COAT_NORMAL )
    float3 mClearCoatNormal;
#endif

#if defined( MATERIAL_HAS_POST_LIGHTING_COLOR )
    float4 mPostLightingColor;
#endif

#if !defined( SHADING_MODEL_CLOTH ) && !defined( SHADING_MODEL_SUBSURFACE ) && !defined( SHADING_MODEL_UNLIT )
#    if defined( MATERIAL_HAS_REFRACTION )
#        if defined( MATERIAL_HAS_ABSORPTION )
    float3 mAbsorption;
#        endif
#        if defined( MATERIAL_HAS_TRANSMISSION )
    float mTransmission;
#        endif
#        if defined( MATERIAL_HAS_IOR )
    float mIor;
#        endif
#        if defined( MATERIAL_HAS_MICRO_THICKNESS ) && ( REFRACTION_TYPE == REFRACTION_TYPE_THIN )
    float mMicroThickness;
#        endif
#    endif
#endif
};

// // clang-format off
// LAYOUT_UNIFORM_BUFFER( CAMERA_PARAMETERS_BIND_POINT, 0 ) __UNIFORM_BUFFER__ CameraParameters 
// { 
//     float mExposure;
//     float mGamma;
// } gCamera;
// // clang-format on

// // Try to be as bindless as possible and bind all available textures andd all materials
// // in one go as an array.
// // clang-format off
// LAYOUT_UNIFORM_BUFFER( MATERIAL_DATA_BIND_POINT, 0 ) __UNIFORM_BUFFER__ ShaderMaterials 
// { 
//     sShaderMaterial mArray[]; 
// } gMaterials;
// // clang-format on

// LAYOUT_UNIFORM( MATERIAL_TEXTURES_BIND_POINT, 0 ) __UNIFORM__ sampler2D gTextures[];

// #if !defined( SHADING_MODEL_UNLIT )
// // clang-format off
// LAYOUT_UNIFORM_BUFFER( DIRECTIONAL_LIGHTS_BIND_POINT, 0 ) __UNIFORM_BUFFER__ DirectionalLights
// {
//     sDirectionalLight mArray[];
// } gDirectionalLights;

// LAYOUT_UNIFORM_BUFFER( PUNCTUAL_LIGHTS_BIND_POINT, 0 ) __UNIFORM_BUFFER__ PointLights
// {
//     sPointLight mArray[];
// } gPointLights;
// // clang-format on
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

#if defined( MATERIAL_HAS_UV0 )
float ColorTextureFetch( int aTexID, int aUVChannel )
{
#    if !defined( MATERIAL_HAS_UV1 )
    return SRGBtoLINEAR( texture( gTextures[aTexID], inUV ) );
#    else
    return SRGBtoLINEAR( texture( gTextures[aTexID], ( aUVChannel == 0 ) ? inUV.xy : inUV.zw ) );
#    endif
}
#endif

#if defined( MATERIAL_HAS_UV0 )
float TextureFetch( int aTexID, int aUVChannel )
{
#    if !defined( MATERIAL_HAS_UV1 )
    return texture( gTextures[aTexID], inUV );
#    else
    return texture( gTextures[aTexID], ( aUVChannel == 0 ) ? inUV.xy : inUV.zw );
#    endif
}
#endif

inline sShaderMaterial GetMaterialData()
{
    if( gMaterialID.mMaterialID > -1 )
        return gMaterials.mArray[gMaterialID.mMaterialID];

    sShaderMaterial lDefault;
    return lDefault;
}

float4 GetBaseColor()
{
    float4 lBaseColor = GetMaterialData().mBaseColorFactor;

#if defined( MATERIAL_HAS_BASE_COLOR_TEXTURE ) && defined( MATERIAL_HAS_UV0 )
    lBaseColor *= ColorTextureFetch( GetMaterialData().mBaseColorTextureID, GetMaterialData().mBaseColorUVChannel );
#endif

    return lBaseColor;
}

float3 GetEmissive()
{
    float3 lEmissive = GetMaterialData().mEmissiveFactor;

#if defined( MATERIAL_HAS_EMISSIVE_TEXTURE ) && defined( MATERIAL_HAS_UV0 )
    lBaseColor *= ColorTextureFetch( GetMaterialData().mEmissiveTextureID, GetMaterialData().mEmissiveUVChannel );
#endif

    return lEmissive;
}

float3 GetNormal()
{
#if defined( MATERIAL_HAS_NORMALS_TEXTURE ) && defined( MATERIAL_HAS_UV0 )
    return GetNormalFromMap( GetMaterialData().mNormalTextureID, GetMaterialData().mNormalUVChannel );
#else
    return normalize( inNormal );
#endif
}

float GetAmbientOcclusion()
{
#if defined( MATERIAL_HAS_OCCLUSION_TEXTURE ) && defined( MATERIAL_HAS_UV0 )
    return TextureFetch( GetMaterialData().mOcclusionTextureID, GetMaterialData().mOcclusionUVChannel );
#else
    return 1.0;
#endif
}

void InitializeMaterial( out MaterialInput aMaterial )
{
    aMaterial.mBaseColor = GetBaseColor();

#if defined( MATERIAL_HAS_NORMALS )
    float3 mNormal = GetNormal();
#endif

#if !defined( SHADING_MODEL_UNLIT )
    aMaterial.mRoughness = clamp( GetMaterialData().mRoughnessFactor, MIN_ROUGHNESS, 1.0 );

#    if defined( MATERIAL_HAS_METAL_ROUGH_TEXTURE )
    float4 lSampledValues = GetAOMetalRough();
    aMaterial.mRoughness *= lSampledValues.g;
#    endif

#    if !defined( SHADING_MODEL_CLOTH )
    aMaterial.mMetallic = clamp( GetMaterialData().mMetallicFactor, 0.0, 1.0 );
#        if defined( MATERIAL_HAS_METAL_ROUGH_TEXTURE )
    aMaterial.mMetallic *= lSampledValues.r;
#        endif
    aMaterial.mReflectance = 0.0f;
#    endif
    aMaterial.mAmbientOcclusion = GetAmbientOcclusion();
#endif

#if defined( MATERIAL_IS_EMISSIVE )
    aMaterial.mEmissive = GetEmissive();
#endif

#if !defined( SHADING_MODEL_CLOTH ) && !defined( SHADING_MODEL_SUBSURFACE ) && !defined( SHADING_MODEL_UNLIT )
    aMaterial.mSheenColor     = float3( 0.0f );
    aMaterial.mSheenRoughness = float3( 0.0f );
#endif

#if defined( MATERIAL_HAS_CLEARCOAT )
    aMaterial.mClearCoat          = 1.0;
    aMaterial.mClearCoatRoughness = 0.0f;
#endif

#if defined( MATERIAL_HAS_ANIROTROPY )
    aMaterial.mAnisotropy          = 0.0;
    aMaterial.mAnisotropyDirection = float3( 1.0, 0.0, 0.0 );
#endif

#if defined( SHADING_MODEL_SUBSURFACE ) || defined( MATERIAL_HAS_REFRACTION )
    aMaterial.mThickness = 0.5;
#endif

#if defined( SHADING_MODEL_SUBSURFACE )
    aMaterial.mSubsurfaceColor = 12.234;
    aMaterial.mSubsurfacePower = float3( 1.0 );
#endif

#if defined( SHADING_MODEL_CLOTH )
    aMaterial.mSheenColor = float3( 1.0 );
#    if defined( MATERIAL_HAS_SUBSURFACE_COLOR )
    aMaterial.mSubsurfaceColor = float3( 0.0 );
#    endif
#endif

#if defined( MATERIAL_HAS_BENT_NORMAL )
    aMaterial.mBentNormal = float3( 0.0, 0.0, 1.0 );
#endif

#if defined( MATERIAL_HAS_CLEAR_COAT ) && defined( MATERIAL_HAS_CLEAR_COAT_NORMAL )
    aMaterial.mClearCoatNormal = float3( 0.0, 0.0, 1.0 );
#endif

#if defined( MATERIAL_HAS_POST_LIGHTING_COLOR )
    aMaterial.mPostLightingColor;
#endif

#if !defined( SHADING_MODEL_CLOTH ) && !defined( SHADING_MODEL_SUBSURFACE ) && !defined( SHADING_MODEL_UNLIT )
#    if defined( MATERIAL_HAS_REFRACTION )
#        if defined( MATERIAL_HAS_ABSORPTION )
    aMaterial.mAbsorption = float3( 0.0 );
#        endif
#        if defined( MATERIAL_HAS_TRANSMISSION )

    aMaterial.mTransmission = 1.0f;
#        endif
#        if defined( MATERIAL_HAS_IOR )

    aMaterial.mIor = 1.5;
#        endif
#        if defined( MATERIAL_HAS_MICRO_THICKNESS ) && ( REFRACTION_TYPE == REFRACTION_TYPE_THIN )
    aMaterial.mMicroThickness = 0.0f;
#        endif
#    endif
#endif
}
void material( out MaterialInput aMaterial ) {}
#if defined( __cplusplus )
#    include "Common/Definitions.h"
#    include "Material.hpp"
#    include "Varying.hpp"
#endif

LAYOUT_LOCATION( 0 ) __SHADER_OUTPUT__ float4 outColor;

#if defined( __cplusplus )
void material( out MaterialInput aMaterial )
{
}
#endif

void main()
{
    MaterialInputs lMaterial;
    InitializeMaterial( lMaterial );

    material( lMaterial );

#if defined( MATERIAL_HAS_EMISSIVE )
    outColor = tonemap( lMaterial.mBaseColor + lMaterial.mEmissive );
#else
    outColor = lMaterial.mBaseColor;
#endif
}
