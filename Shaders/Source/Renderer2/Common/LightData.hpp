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