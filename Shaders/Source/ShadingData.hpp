#ifndef _SHADING_DATA_H_
#define _SHADING_DATA_H_

#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#endif

struct ShadingData
{
    // The material's diffuse color, as derived from baseColor and metallic.
    // This color is pre-multiplied by alpha and in the linear sRGB color space.
    float3 mDiffuseColor;

    // The material's specular color, as derived from baseColor and metallic.
    // This color is pre-multiplied by alpha and in the linear sRGB color space.
    float3 mF0;

    // The perceptual roughness is the roughness value set in MaterialInputs,
    // with extra processing:
    // - Clamped to safe values
    // - Filtered if specularAntiAliasing is enabled
    // This value is between 0.0 and 1.0.
    float mPerceptualRoughness;

    // The roughness value expected by BRDFs. This value is the square of
    // perceptualRoughness. This value is between 0.0 and 1.0.
    float mRoughness;

    // Set to 1.0 for now until we find a use for it.
    float mEnergyCompensation;

#if defined( MATERIAL_HAS_SUBSURFACE_COLOR )
    float3 mSubsurfaceColor;
#endif

#if defined( MATERIAL_HAS_SHEEN_COLOR ) && !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE)
    float3 mSheenColor;
    float mSheenPerceptualRoughness;
    float mSheenRoughness;
    float mSheenScaling;
#endif
};

#endif