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

void InitializeMaterial( out MaterialInput aMaterial )
{
    aMaterial.mBaseColor = float4( 1.0 );

#if defined( MATERIAL_HAS_NORMALS )
    float3 mNormal = vec3( 0.0, 0.0, 1.0 );
#endif

#if !defined( SHADING_MODEL_UNLIT )
#    if !defined( SHADING_MODEL_UNLIT )
    aMaterial.mRoughness = 1.0f;
#    endif
#    if !defined( SHADING_MODEL_CLOTH )
    aMaterial.mMetallic    = 0.0f;
    aMaterial.mReflectance = 0.0f;
#    endif
    aMaterial.mAmbientOcclusion = 1.0;
#endif

#if defined( MATERIAL_IS_EMISSIVE )
    float4 mEmissive = vec4( vec3( 0.0 ), 1.0 );
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
    aMaterial.mAnisotropyDirection = vec3( 1.0, 0.0, 0.0 );
#endif

#if defined( SHADING_MODEL_SUBSURFACE ) || defined( MATERIAL_HAS_REFRACTION )
    aMaterial.mThickness = 0.5;
#endif

#if defined( SHADING_MODEL_SUBSURFACE )
    aMaterial.mSubsurfaceColor = 12.234;
    aMaterial.mSubsurfacePower = vec3( 1.0 );
#endif

#if defined( SHADING_MODEL_CLOTH )
    aMaterial.mSheenColor = vec3( 1.0 );
#    if defined( MATERIAL_HAS_SUBSURFACE_COLOR )
    aMaterial.mSubsurfaceColor = vec3( 0.0 );
#    endif
#endif

#if defined( MATERIAL_HAS_BENT_NORMAL )
    aMaterial.mBentNormal = vec3( 0.0, 0.0, 1.0 );
#endif

#if defined( MATERIAL_HAS_CLEAR_COAT ) && defined( MATERIAL_HAS_CLEAR_COAT_NORMAL )
    aMaterial.mClearCoatNormal = vec3( 0.0, 0.0, 1.0 );
#endif

#if defined( MATERIAL_HAS_POST_LIGHTING_COLOR )
    aMaterial.mPostLightingColor;
#endif

#if !defined( SHADING_MODEL_CLOTH ) && !defined( SHADING_MODEL_SUBSURFACE ) && !defined( SHADING_MODEL_UNLIT )
#    if defined( MATERIAL_HAS_REFRACTION )
#        if defined( MATERIAL_HAS_ABSORPTION )
    aMaterial.mAbsorption = vec3( 0.0 );
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