#ifndef _LIGHT_DATA_HPP_
#define _LIGHT_DATA_HPP_

#if defined( __cplusplus )
#    include "Definitions.hpp"
#endif

// Shared with engine renderer code

struct sDirectionalLight
{
    float4 mColorIntensity;
    float3 mDirection;
    int mCastsShadows;
};

struct sPunctualLight
{
    float4 mColorIntensity;
    float3 mPosition;
    int mCastsShadows;
};

#endif