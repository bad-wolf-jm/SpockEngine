#ifndef _SHADOWING_H_
#define _SHADOWING_H_

#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#    include "Varying.hpp"
#endif

float Shadow(bool aIsDirectional, sampler2D aShadowMap)
{
    return 1.0;
}

#endif