#ifndef _LIGHT_DATA_HPP_
#define _LIGHT_DATA_HPP_

#if defined( __cplusplus )
#    include "Definitions.hpp"
#endif

// Shared with engine renderer code

struct ALIGN( 16 ) sDirectionalLight
{
    float4 mColorIntensity;
    float3 mDirection;
    ALIGN( 16 ) bool mCastsShadows;
};

struct ALIGN( 16 ) sPunctualLight
{
    float4 mColorIntensity;
    float3 mPosition;
    ALIGN( 16 ) bool mCastsShadows;
};

#endif