#ifndef _LIGHT_DATA_H_
#define _LIGHT_DATA_H_

#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#endif

struct LightData
{
    // The color (.rgb) and pre-exposed intensity (.w) of the light. The color is an RGB value in the linear sRGB color space.
    // The pre-exposed intensity is the intensity of the light multiplied by the camera's exposure value.
    float4 mColorIntensity;

    // The normalized light vector, in world space (direction from the current fragment's position to the light).
    float3 mL;

    // The normalized light half vector, in world space (direction from the current fragment's position to the light).
    float3 mH;

    // The dot product of the shading normal (with normal mapping applied) and the light vector. This value is equal to the result of
    // saturate(dot(getWorldSpaceNormal(), lightData.l)). This value is always between 0.0 and 1.0. When the value is <= 0.0,
    // the current fragment is not visible from the light and lighting computations can be skipped.
    float mNdotL;

    // The position of the light in world space.
    float3 mWorldPosition;

    // Attenuation of the light based on the distance from the current fragment to the light in world space. This value between 0.0
    // and 1.0 is computed differently for each type of light (it's always 1.0 for directional lights).
    float mAttenuation;

    // Visibility factor computed from shadow maps or other occlusion data specific to the light being evaluated. This value is between
    // 0.0 and 1.0.
    float mVisibility;
};
#endif