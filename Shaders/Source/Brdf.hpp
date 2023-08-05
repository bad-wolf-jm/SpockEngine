#ifndef _BRDF_H_
#define _BRDF_H_

#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#endif

float D_GGX( float roughness, float NoH, const float3 h )
{
    // Walter et al. 2007, "Microfacet Models for Refraction through Rough Surfaces"

    float oneMinusNoHSquared = 1.0 - NoH * NoH;

    float a = NoH * roughness;
    float k = roughness / ( oneMinusNoHSquared + a * a );
    float d = k * k * ( 1.0 / PI );

    return saturate( d );
}

float V_SmithGGXCorrelated( float roughness, float NoV, float NoL )
{
    // Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"

    float a2 = roughness * roughness;

    float lambdaV = NoL * sqrt( ( NoV - a2 * NoV ) * NoV + a2 );
    float lambdaL = NoV * sqrt( ( NoL - a2 * NoL ) * NoL + a2 );
    float v       = 0.5 / ( lambdaV + lambdaL );

    return saturate( v );
}

float3 F_Schlick( const float3 f0, float f90, float VoH )
{
    // Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"
    return f0 + ( f90 - f0 ) * pow5( 1.0 - VoH );
}

float F_Schlick( float f0, float f90, float VoH )
{
    return f0 + ( f90 - f0 ) * pow5( 1.0 - VoH );
}
#endif