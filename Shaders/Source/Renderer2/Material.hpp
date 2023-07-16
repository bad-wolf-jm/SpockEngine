#if defined( __cplusplus )
#    include "Common/Definitions.h"
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

struct sShaderMaterial
{
    float4 mBaseColorFactor;
    float  mMetallicFactor;
    float  mRoughnessFactor;
    float  mOcclusionStrength;
    float4 mEmissiveFactor;
#if defined( MATERIAL_HAS_UV1 )
    int mBaseColorUVChannel;
    int mEmissiveUVChannel;
    int mNormalUVChannel;
    int mMetalnessUVChannel;
    int mOcclusionUVChannel;
#endif
};

#if defined( MATERIAL_HAS_BASE_COLOR_TEXTURE )
layout( set = 0, location = BASE_COLOR_TEXTURE_BIND_POINT ) uniform sampler2D gBaseColorTexture;
#endif

#if defined( MATERIAL_HAS_METAL_ROUGH_TEXTURE )
layout( set = 0, location = METAL_ROUGH_TEXTURE_BIND_POINT ) uniform sampler2D gAOMetalRoughTexture;
#endif

#if defined( MATERIAL_HAS_AO_TEXTURE )
layout( set = 0, location = AO_TEXTURE_BIND_POINT ) uniform sampler2D gAOTexture;
#endif

#if defined( MATERIAL_HAS_EMISSIVE_TEXTURE )
layout( set = 0, location = EMISSIVE_TEXTURE_BIND_POINT ) uniform sampler2D gEmissiveTexture;
#endif

#if defined( MATERIAL_HAS_HORMALS_TEXTURE )
layout( set = 0, location = NORMALS_TEXTURE_BIND_POINT ) uniform sampler2D gNormalsTexture;
#endif


float4 GetBaseColor()
{
    float4 lBaseColor = gMaterialData.mBaseColorFactor;

#if defined( MATERIAL_HAS_BASE_COLOR_TEXTURE )
#    if defined( MATERIAL_HAS_UV0 ) && !defined( MATERIAL_HAS_UV1 )
    lBaseColor *= SRGBtoLINEAR( texture( gBaseColorTexture, inUV ) );
#    else
    lBaseColor *= SRGBtoLINEAR( texture( gBaseColorTexture, ( gMaterialData.mBaseColorUVChannel == 0 ) ? inUV.xy : inUV.zw ) );
#    endif
#endif

    return lBaseColor;
}

float3 GetEmissive()
{
    float3 lEmissive = gMaterialData.mEmissiveFactor;

#if defined( MATERIAL_HAS_EMISSIVE_TEXTURE )
#    if defined( MATERIAL_HAS_UV0 ) && !defined( MATERIAL_HAS_UV1 )
    lEmissive *= SRGBtoLINEAR( texture( gEmissiveTexture, inUV ) );
#    else
    lEmissive *= SRGBtoLINEAR( texture( gEmissiveTexture, ( gMaterialData.mEmissiveUVChannel == 0 ) ? inUV.xy : inUV.zw ) );
#    endif
#endif

    return lEmissive;
}

float3 GetNormal()
{
#if defined( MATERIAL_HAS_NORMAL_TEXTURE )
#    if defined( MATERIAL_HAS_UV0 ) && !defined( MATERIAL_HAS_UV1 )
    return GetNormalFomMap( gNormalTexture, inUV );
#    else
    return GetNormalFomMap( gNormalTexture, ( gMaterialData.mNormalUVChannel == 0 ) ? inUV.xy : inUV.zw );
#    endif
#else
    return normalize( inNormal );
#endif
}

float GetAmbientOcclusion()
{
#if defined( MATERIAL_HAS_NORMAL_TEXTURE )
#    if defined( MATERIAL_HAS_UV0 ) && !defined( MATERIAL_HAS_UV1 )
    return texture( gAOTexture, inUV );
#    else
    return texture( gAOTexture, ( gMaterialData.mAOUVChannel == 0 ) ? inUV.xy : inUV.zw );
#    endif
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
    aMaterial.mRoughness = clamp( gMaterialData.mRoughnessFactor, MIN_ROUGHNESS, 1.0 );

#    if defined( MATERIAL_HAS_METAL_ROUGH_TEXTURE )
    float4 lSampledValues = GetAOMetalRough();
    aMaterial.mRoughness *= lSampledValues.g;
#    endif

#    if !defined( SHADING_MODEL_CLOTH )
    aMaterial.mMetallic = clamp( gMaterialData.mMetallicFactor, 0.0, 1.0 );
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