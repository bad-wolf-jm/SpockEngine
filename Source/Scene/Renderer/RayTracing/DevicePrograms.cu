#include "Core/CUDA/CudaAssert.h"

#include <cuda_runtime.h>
#include <optix_device.h>

#include "Core/CUDA/Random.h"
#include "LaunchParams.h"

#define NUM_LIGHT_SAMPLES 8

namespace SE::Core
{

    typedef Cuda::LCG<16> Random;

    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    extern "C" __constant__ sLaunchParams optixLaunchParams;

    /*! per-ray data now captures random number generator, so programs
        can access RNG state */
    struct PRD
    {
        Random     mRandom;
        math::vec3 mPixelColor;
        math::vec3 mPixelNormal;
        math::vec3 mPixelAlbedo;
    };

    static SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF float clampf( float f, float aMin, float aMax )
    {
        return min( aMax, max( aMin, f ) );
    }

    static SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF void *unpackPointer( uint32_t i0, uint32_t i1 )
    {
        const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
        void          *ptr  = reinterpret_cast<void *>( uptr );
        return ptr;
    }

    static SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF void packPointer( void *ptr, uint32_t &i0, uint32_t &i1 )
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
        i0                  = uptr >> 32;
        i1                  = uptr & 0x00000000ffffffff;
    }

    template <typename T>
    static SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF T *GetPerRayData()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T *>( unpackPointer( u0, u1 ) );
    }

    extern "C" CUDA_KERNEL_DEFINITION void __closesthit__shadow()
    { /* not going to be used ... */
    }

    static SE_CUDA_DEVICE_FUNCTION_DEF void GetPrimitive( sTriangleMeshSBTData const &aSbtData, int aPrimitiveID, VertexData &aV1,
                                                          VertexData &aV2, VertexData &aV3 )
    {
        uint32_t const     lVertexOffset = aSbtData.mVertexOffset;
        math::uvec3 const &lPrimitive    = optixLaunchParams.mIndexBuffer[aPrimitiveID];

        aV1 = aSbtData.mVertexBuffer[lVertexOffset + lPrimitive.x];
        aV2 = aSbtData.mVertexBuffer[lVertexOffset + lPrimitive.y];
        aV3 = aSbtData.mVertexBuffer[lVertexOffset + lPrimitive.z];
    }

    math::vec3 SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF GetNormalFromMap( math::vec3 aInNormal, cudaTextureObject_t aNormalSampler,
                                                                            math::vec2 aCoords )
    {
        // // Perturb normal, see http://www.thetenthplanet.de/archives/1180
        // auto       lNormalMapSample = tex2D<float4>( aNormalSampler, aCoords.x, aCoordx.y );
        // math::vec3 lNormalMapValue  = math::vec3{ lNormalMapSample.x, lNormalMapSample.y, lNormalMapSample.z };

        // math::vec3 lTangentNormal = normalize( lNormalMapValue * 2.0 - vec3( 1.0 ) );

        // math::vec3 dp1  = dFdx( inWorldPos );
        // math::vec3 dp2  = dFdy( inWorldPos );
        // math::vec2 duv1 = dFdx( aCoords );
        // math::vec2 duv2 = dFdy( aCoords );

        // // solve the linear system
        // math::vec3 dp1perp = cross( aInNormal, dp1 );
        // math::vec3 dp2perp = cross( dp2, aInNormal );
        // math::vec3 T       = dp2perp * duv1.x + dp1perp * duv2.x;
        // math::vec3 B       = dp2perp * duv1.y + dp1perp * duv2.y;

        // // construct a scale-invariant frame
        // float invmax = inversesqrt( max( dot( T, T ), dot( B, B ) ) );

        // return normalize( math::mat3( T * invmax, B * invmax, aInNormal ) * lTangentNormal );

        return normalize( aInNormal );
    }

    constexpr float PI = 3.14159265359;

    static SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF float DistributionGGX( vec3 N, vec3 H, float roughness )
    {
        float a      = roughness * roughness;
        float a2     = a * a;
        float NdotH  = max( dot( N, H ), 0.0 );
        float NdotH2 = NdotH * NdotH;

        float nom   = a2;
        float denom = ( NdotH2 * ( a2 - 1.0 ) + 1.0 );
        denom       = PI * denom * denom;

        return nom / denom;
    }

    static SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF float GeometrySchlickGGX( float NdotV, float roughness )
    {
        float r     = ( roughness + 1.0 );
        float k     = ( r * r ) / 8.0;
        float nom   = NdotV;
        float denom = NdotV * ( 1.0 - k ) + k;

        return nom / denom;
    }

    static SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF float GeometrySmith( vec3 N, vec3 V, vec3 L, float roughness )
    {
        float NdotV = max( dot( N, V ), 0.0 );
        float NdotL = max( dot( N, L ), 0.0 );
        float ggx2  = GeometrySchlickGGX( NdotV, roughness );
        float ggx1  = GeometrySchlickGGX( NdotL, roughness );

        return ggx1 * ggx2;
    }

    static SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF math::vec3 FresnelSchlick( float cosTheta, math::vec3 F0 )
    {
        return F0 + ( math::vec3( 1.0 ) - F0 ) * pow( clampf( 1.0f - cosTheta, 0.0f, 1.0f ), 5.0f );
    }

    static SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF math::vec3 CookTorrance( math::vec3 F0, math::vec3 N, math::vec3 L, math::vec3 V,
                                                                               math::vec3 H, float roughness )
    {
        float NDF = DistributionGGX( N, H, roughness );
        float G   = GeometrySmith( N, V, L, roughness );
        vec3  F   = FresnelSchlick( max( dot( H, V ), 0.0f ), F0 );
        return ( NDF * G * F ) / ( 4 * max( dot( N, V ), 0.0f ) * max( dot( N, L ), 0.0f ) + 0.0001f );
    }

    static SE_CUDA_INLINE SE_CUDA_DEVICE_FUNCTION_DEF math::vec3
    ComputeLightContribution( math::vec3 aBaseColor, math::vec3 aSurfaceNormal, math::vec3 aEyeDirection, math::vec3 aLightDirection,
                              math::vec3 aRadiance, float aMetal, float aRough )
    {
        // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
        // of 0.04 and if it's a metal, use the base color as F0 (metallic workflow)
        math::vec3 lF0 = math::mix( math::vec3( 0.04 ), aBaseColor, aMetal );

        math::vec3 H = normalize( aEyeDirection + aLightDirection );

        // Cook-Torrance BRDF
        math::vec3 lSpecular = CookTorrance( lF0, aSurfaceNormal, aLightDirection, aEyeDirection, H, aRough );

        // kS is equal to Fresnel
        math::vec3 kS = FresnelSchlick( max( dot( H, aEyeDirection ), 0.0 ), lF0 );

        // for energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve
        // this relationship the diffuse component (kD) should equal 1.0 - kS.
        math::vec3 kD = vec3( 1.0 ) - kS;

        // multiply kD by the inverse metalness such that only non-metals have diffuse lighting, or a linear blend if partly metal
        // (pure metals have no diffuse light).
        kD *= ( 1.0 - aMetal );

        // scale light by NdotL
        float NdotL = max( dot( aSurfaceNormal, aLightDirection ), 0.0 );

        // add to outgoing radiance Lo
        return ( kD * aBaseColor / PI + lSpecular ) * aRadiance * NdotL;
    }

    extern "C" CUDA_KERNEL_DEFINITION void __closesthit__radiance()
    {
        const sTriangleMeshSBTData &sbtData = *(const sTriangleMeshSBTData *)optixGetSbtDataPointer();
        PRD                        &prd     = *GetPerRayData<PRD>();

        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        const int lPrimitiveID = optixGetPrimitiveIndex() + sbtData.mIndexOffset;

        VertexData lV1, lV2, lV3;
        GetPrimitive( sbtData, lPrimitiveID, lV1, lV2, lV3 );

        const math::vec3 &A  = lV1.Position;
        const math::vec3 &B  = lV2.Position;
        const math::vec3 &C  = lV3.Position;
        math::vec3        Ng = glm::cross( B - A, C - A );

        const math::vec3 &NA = lV1.Normal;
        const math::vec3 &NB = lV2.Normal;
        const math::vec3 &NC = lV3.Normal;
        math::vec3        Ns = ( ( 1.f - u - v ) * NA + u * NB + v * NC );

        float3           lWorldRayDirection = optixGetWorldRayDirection();
        const math::vec3 rayDir             = { lWorldRayDirection.x, lWorldRayDirection.y, lWorldRayDirection.z };

        if( dot( rayDir, Ng ) > 0.f ) Ng = -Ng;
        Ng = normalize( Ng );

        if( dot( Ng, Ns ) < 0.f ) Ns -= 2.f * dot( Ng, Ns ) * Ng;
        Ns = normalize( Ns );

        const math::vec2 &TA = lV1.TexCoords_0;
        const math::vec2 &TB = lV2.TexCoords_0;
        const math::vec2 &TC = lV3.TexCoords_0;

        const math::vec2 tc = ( 1.f - u - v ) * TA + u * TB + v * TC;

        auto lMaterialData = optixLaunchParams.mMaterials[sbtData.mMaterialID];
        auto lBaseColor    = optixLaunchParams.mTextures[lMaterialData.mBaseColorTextureID];
        auto lTextureValue = tex2D<float4>( lBaseColor.mTextureObject, tc.x, tc.y );

        math::vec3 diffuseColor =
            math::vec3{ lTextureValue.x, lTextureValue.y, lTextureValue.z } * math::vec3( lMaterialData.mBaseColorFactor );

        // start with some ambient term
        math::vec3 pixelColor = ( 0.1f + 0.2f * fabsf( dot( Ns, rayDir ) ) ) * diffuseColor;

        math::vec3 lTNorm;
        if( lMaterialData.mNormalTextureID == 0 )
            lTNorm = normalize( Ns );
        else
            lTNorm = GetNormalFromMap( Ns, optixLaunchParams.mTextures[lMaterialData.mNormalTextureID].mTextureObject, tc );

        const float cMinRoughness = 0.04;
        auto lMetalRough = tex2D<float4>( optixLaunchParams.mTextures[lMaterialData.mMetalnessTextureID].mTextureObject, tc.x, tc.y );
        const float lMetallic  = lMetalRough.x * clampf( lMaterialData.mMetallicFactor, 0.0, 1.0 );
        const float lRoughness = lMetalRough.y * clampf( lMaterialData.mRoughnessFactor, cMinRoughness, 1.0 );

        // // ------------------------------------------------------------------
        // // compute shadow
        // // ------------------------------------------------------------------
        // const math::vec3 surfPos = ( 1.f - u - v ) * A + u * B + v * C;

        // const int numLightSamples = optixLaunchParams.mNumLightSamples;
        // for( int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++ )
        // {
        //     // produce random light sample
        //     const math::vec3 lightPos = optixLaunchParams.mLight.mOrigin + prd.mRandom() * optixLaunchParams.mLight.mDu +
        //                                 prd.mRandom() * optixLaunchParams.mLight.mDv;
        //     math::vec3 lightDir  = lightPos - surfPos;
        //     float      lightDist = glm::length( lightDir );
        //     lightDir             = glm::normalize( lightDir );

        //     // trace shadow ray:
        //     const float NdotL = dot( lightDir, Ns );
        //     if( NdotL >= 0.f )
        //     {
        //         math::vec3 lightVisibility( 0.0f );
        //         // the values we store the PRD pointer in:
        //         uint32_t u0, u1;
        //         packPointer( &lightVisibility, u0, u1 );
        //         auto lRayOrigin = surfPos + 1e-3f * Ng;
        //         optixTrace( optixLaunchParams.mSceneRoot, float3{ lRayOrigin.x, lRayOrigin.y, lRayOrigin.z },
        //                     float3{ lightDir.x, lightDir.y, lightDir.z },
        //                     1e-3f,
        //                     lightDist * ( 1.f - 1e-3f ),
        //                     0.0f,
        //                     OptixVisibilityMask( 255 ),
        //                     OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
        //                     OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE, u0, u1 );
        //         pixelColor += lightVisibility * optixLaunchParams.mLight.mPower * diffuseColor *
        //                       ( NdotL / ( lightDist * lightDist * numLightSamples ) );
        //     }
        // }

        prd.mPixelNormal = Ns;
        prd.mPixelAlbedo = diffuseColor;
        prd.mPixelColor  = pixelColor;
    }

    extern "C" CUDA_KERNEL_DEFINITION void __anyhit__radiance() {}

    extern "C" CUDA_KERNEL_DEFINITION void __anyhit__shadow() {}

    extern "C" CUDA_KERNEL_DEFINITION void __miss__radiance()
    {
        PRD &prd = *GetPerRayData<PRD>();
        // set to constant white as background color
        prd.mPixelColor = math::vec3( 1.f );
    }

    extern "C" CUDA_KERNEL_DEFINITION void __miss__shadow()
    {
        // we didn't hit anything, so the light is visible
        math::vec3 &prd = *(math::vec3 *)GetPerRayData<math::vec3>();
        prd             = math::vec3( 1.f );
    }

    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" CUDA_KERNEL_DEFINITION void __raygen__renderFrame()
    {
        // compute a test pattern based on pixel ID
        const int   ix     = optixGetLaunchIndex().x;
        const int   iy     = optixGetLaunchIndex().y;
        const auto &camera = optixLaunchParams.mCamera;

        PRD prd;
        prd.mRandom.init( ix + optixLaunchParams.mFrame.mSize.x * iy, optixLaunchParams.mFrame.mFrameID );
        prd.mPixelColor = math::vec3( 0.f );

        // the values we store the PRD pointer in:
        uint32_t u0, u1;
        packPointer( &prd, u0, u1 );

        int numPixelSamples = optixLaunchParams.mNumPixelSamples;

        math::vec3 pixelColor( 0.f );
        math::vec3 pixelNormal( 0.f );
        math::vec3 pixelAlbedo( 0.f );
        for( int sampleID = 0; sampleID < numPixelSamples; sampleID++ )
        {
            math::vec2 screen( math::vec2( ix + prd.mRandom(), iy + prd.mRandom() ) / math::vec2( optixLaunchParams.mFrame.mSize ) );

            math::vec3 rayDir =
                normalize( camera.mDirection + ( screen.x - 0.5f ) * camera.mHorizontal + ( screen.y - 0.5f ) * camera.mVertical );

            optixTrace( optixLaunchParams.mSceneRoot, float3{ camera.mPosition.x, camera.mPosition.y, camera.mPosition.z },
                        float3{ rayDir.x, rayDir.y, rayDir.z }, 0.f, 1e20f, 0.0f, OptixVisibilityMask( 255 ),
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT, RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE, u0, u1 );

            pixelColor += prd.mPixelColor;
            pixelNormal += prd.mPixelNormal;
            pixelAlbedo += prd.mPixelAlbedo;
        }

        math::vec4 rgba( pixelColor / static_cast<float>( numPixelSamples ), 1.f );
        math::vec4 albedo( pixelAlbedo / static_cast<float>( numPixelSamples ), 1.f );
        math::vec4 normal( pixelNormal / static_cast<float>( numPixelSamples ), 1.f );

        // and write/accumulate to frame buffer ...
        const uint32_t fbIndex = ix + iy * optixLaunchParams.mFrame.mSize.x;
        if( optixLaunchParams.mFrame.mFrameID > 0 )
        {
            rgba += float( optixLaunchParams.mFrame.mFrameID ) * math::vec4( optixLaunchParams.mFrame.mColorBuffer[fbIndex] );
            rgba /= ( optixLaunchParams.mFrame.mFrameID + 1.f );
        }
        optixLaunchParams.mFrame.mColorBuffer[fbIndex]  = (math::vec4)rgba;
        optixLaunchParams.mFrame.mAlbedoBuffer[fbIndex] = (math::vec4)albedo;
        optixLaunchParams.mFrame.mNormalBuffer[fbIndex] = (math::vec4)normal;
    }

} // namespace SE::Core
