#ifndef _SHADER_MATERIAL_H_
#define _SHADER_MATERIAL_H_

#if defined( __cplusplus )
#    include "Definitions.hpp"
#endif


struct ALIGN( 16 ) sShaderMaterial
{
    float4 mBaseColorFactor;
    float  mMetallicFactor;
    float  mRoughnessFactor;
    float  mOcclusionStrength;
    ALIGN( 16 ) float4 mEmissiveFactor;
    int mBaseColorUVChannel;
    int mBaseColorTextureID;
    int mEmissiveUVChannel;
    int mEmissiveTextureID;

    int mNormalUVChannel;
    int mNormalTextureID;
    int mMetalnessUVChannel;
    int mMetalnessTextureID;

    int mOcclusionUVChannel;
    int mOcclusionTextureID;
};

#endif