

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

    float mRoughness;
#if !defined( SHADING_MODEL_CLOTH )
    float mMetallic;
    float mReflectance;
#endif
    float mAmbientOcclusion;

#if defined( MATERIAL_IS_EMISSIVE )
    float4 mEmissive;
#endif

#if defined( MATERIAL_HAS_SHEEN_COLOR )
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

#if !defined( SHADING_MODEL_CLOTH ) && !defined( SHADING_MODEL_SUBSURFACE )
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

#if defined( MATERIAL_HAS_UV0 )
float4 ColorTextureFetch( int aTexID, int aUVChannel )
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
    return texture( gTextures[aTexID], inUV ).r;
#    else
    return texture( gTextures[aTexID], ( aUVChannel == 0 ) ? inUV.xy : inUV.zw ).r;
#    endif
}
#endif

sShaderMaterial GetMaterialData()
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
    float3 lEmissive = GetMaterialData().mEmissiveFactor.xyz;

#if defined( MATERIAL_HAS_EMISSIVE_TEXTURE ) && defined( MATERIAL_HAS_UV0 )
    lEmissive *= ColorTextureFetch( GetMaterialData().mEmissiveTextureID, GetMaterialData().mEmissiveUVChannel ).xyz;
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

float4 GetAOMetalRough()
{
#if defined( MATERIAL_HAS_METAL_ROUGH_TEXTURE ) && defined( MATERIAL_HAS_UV0 )
    return ColorTextureFetch( GetMaterialData().mMetalnessTextureID, GetMaterialData().mMetalnessUVChannel );
#else
    return vec4( 1.0 );
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

void InitializeMaterial( out MaterialInputs aMaterial )
{
    aMaterial.mBaseColor = GetBaseColor();

#if defined( MATERIAL_HAS_NORMALS )
    aMaterial.mNormal = GetNormal();
#endif
    aMaterial.mRoughness = clamp( GetMaterialData().mRoughnessFactor, 0.0, 1.0 );

#if defined( MATERIAL_HAS_METAL_ROUGH_TEXTURE )
    float4 lSampledValues = GetAOMetalRough();
    aMaterial.mRoughness *= lSampledValues.g;
#endif

#if !defined( SHADING_MODEL_CLOTH )
    aMaterial.mMetallic = clamp( GetMaterialData().mMetallicFactor, 0.0, 1.0 );
#    if defined( MATERIAL_HAS_METAL_ROUGH_TEXTURE )
    aMaterial.mMetallic *= lSampledValues.r;
#    endif
    aMaterial.mReflectance = 0.0f;
#endif
    aMaterial.mAmbientOcclusion = GetAmbientOcclusion();

#if defined( MATERIAL_IS_EMISSIVE )
    aMaterial.mEmissive = GetEmissive();
#endif

#if defined(MATERIAL_HAS_SHEEN_COLOR) && !defined( SHADING_MODEL_CLOTH ) && !defined( SHADING_MODEL_SUBSURFACE )
    aMaterial.mSheenColor     = float3( 0.0f );
    aMaterial.mSheenRoughness = 0.0f;
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

#if !defined( SHADING_MODEL_CLOTH ) && !defined( SHADING_MODEL_SUBSURFACE )
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