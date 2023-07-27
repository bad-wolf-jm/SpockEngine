#ifndef _LIGHT_CALCULATION_H_
#define _LIGHT_CALCULATION_H_

#if defined( __cplusplus )
#    include "Common/Definitions.h"
#    include "Material.hpp"
#    include "Varying.hpp"
#endif


void AddEmissive( MaterialInputs aMaterial, inout float4 aColor )
{
#if defined( MATERIAL_HAS_EMISSIVE )
    vec4  lEmissive    = material.mEmissive;
    float lAttenuation = mix( 1.0, getExposure(), lEmissive.w );
    aColor.rgb += lEmissive.rgb * ( lAttenuation * aColor.a );
#endif
}

float4 EvaluateMaterial( MaterialInputs aMaterial )
{
    float4 lColor = aMaterial.mBaseColor;
    AddEmissive(aMaterial, lColor);

    return lColor;
}

#endif
