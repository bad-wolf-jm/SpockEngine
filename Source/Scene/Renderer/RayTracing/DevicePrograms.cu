#include "Core/Cuda/CudaAssert.h"

#include <cuda_runtime.h>
#include <optix_device.h>

#include "Core/Cuda/Random.h"
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

    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------

    extern "C" CUDA_KERNEL_DEFINITION void __closesthit__shadow()
    { /* not going to be used ... */
    }

    extern "C" CUDA_KERNEL_DEFINITION void __closesthit__radiance()
    {
        const sTriangleMeshSBTData &sbtData = *(const sTriangleMeshSBTData *)optixGetSbtDataPointer();
        PRD                        &prd     = *GetPerRayData<PRD>();

        // ------------------------------------------------------------------
        // gather some basic hit information
        // ------------------------------------------------------------------
        const int         primID = optixGetPrimitiveIndex();
        const math::ivec3 index  = sbtData.mIndex[primID];
        const float       u      = optixGetTriangleBarycentrics().x;
        const float       v      = optixGetTriangleBarycentrics().y;

        // ------------------------------------------------------------------
        // compute normal, using either shading normal (if avail), or
        // geometry normal (fallback)
        // ------------------------------------------------------------------
        // const math::vec3 &A  = sbtData.mVertex[index.x];
        // const math::vec3 &B  = sbtData.mVertex[index.y];
        // const math::vec3 &C  = sbtData.mVertex[index.z];
        const int lPrimitiveID = optixGetPrimitiveIndex() + sbtData.mIndexOffset;

        const math::vec3 &A =
            optixLaunchParams.mVertexBuffer[sbtData.mVertexOffset + optixLaunchParams.mIndexBuffer[lPrimitiveID].x].Position;
        const math::vec3 &B =
            optixLaunchParams.mVertexBuffer[sbtData.mVertexOffset + optixLaunchParams.mIndexBuffer[lPrimitiveID].y].Position;
        const math::vec3 &C =
            optixLaunchParams.mVertexBuffer[sbtData.mVertexOffset + optixLaunchParams.mIndexBuffer[lPrimitiveID].z].Position;

        math::vec3 Ng = glm::cross( B - A, C - A );
        math::vec3 Ns =
            ( sbtData.mNormal )
                ? ( ( 1.f - u - v ) * sbtData.mNormal[index.x] + u * sbtData.mNormal[index.y] + v * sbtData.mNormal[index.z] )
                : Ng;

        // ------------------------------------------------------------------
        // face-forward and normalize normals
        // ------------------------------------------------------------------
        float3           lWorldRayDirection = optixGetWorldRayDirection();
        const math::vec3 rayDir             = { lWorldRayDirection.x, lWorldRayDirection.y, lWorldRayDirection.z };

        if( dot( rayDir, Ng ) > 0.f ) Ng = -Ng;
        Ng = normalize( Ng );

        if( dot( Ng, Ns ) < 0.f ) Ns -= 2.f * dot( Ng, Ns ) * Ng;
        Ns = normalize( Ns );

        math::vec3 diffuseColor = sbtData.mColor;
        if( sbtData.mHasTexture && sbtData.mTexCoord )
        {
            const math::vec2 &A =
                optixLaunchParams.mVertexBuffer[sbtData.mVertexOffset + optixLaunchParams.mIndexBuffer[lPrimitiveID].x].TexCoords_0;
            const math::vec2 &B =
                optixLaunchParams.mVertexBuffer[sbtData.mVertexOffset + optixLaunchParams.mIndexBuffer[lPrimitiveID].y].TexCoords_0;
            const math::vec2 &C =
                optixLaunchParams.mVertexBuffer[sbtData.mVertexOffset + optixLaunchParams.mIndexBuffer[lPrimitiveID].z].TexCoords_0;

            const math::vec2 tc = ( 1.f - u - v ) * A + u * B + v * C;

            auto       lValue      = tex2D<float4>( sbtData.mTexture, tc.x, tc.y );
            math::vec4 fromTexture = { lValue.x, lValue.y, lValue.z, lValue.w };
            diffuseColor *= (math::vec3)fromTexture;
        }

        // start with some ambient term
        math::vec3 pixelColor = ( 0.1f + 0.2f * fabsf( dot( Ns, rayDir ) ) ) * diffuseColor;

        // ------------------------------------------------------------------
        // compute shadow
        // ------------------------------------------------------------------
        const math::vec3 surfPos = ( 1.f - u - v ) * A + u * B + v * C;

        const int numLightSamples = optixLaunchParams.mNumLightSamples;
        for( int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++ )
        {
            // produce random light sample
            const math::vec3 lightPos = optixLaunchParams.mLight.mOrigin + prd.mRandom() * optixLaunchParams.mLight.mDu +
                                        prd.mRandom() * optixLaunchParams.mLight.mDv;
            math::vec3 lightDir  = lightPos - surfPos;
            float      lightDist = glm::length( lightDir );
            lightDir             = glm::normalize( lightDir );

            // trace shadow ray:
            const float NdotL = dot( lightDir, Ns );
            if( NdotL >= 0.f )
            {
                math::vec3 lightVisibility( 0.0f );
                // the values we store the PRD pointer in:
                uint32_t u0, u1;
                packPointer( &lightVisibility, u0, u1 );
                auto lRayOrigin = surfPos + 1e-3f * Ng;
                optixTrace( optixLaunchParams.mSceneRoot, float3{ lRayOrigin.x, lRayOrigin.y, lRayOrigin.z },
                            float3{ lightDir.x, lightDir.y, lightDir.z },
                            1e-3f,                       
                            lightDist * ( 1.f - 1e-3f ), 
                            0.0f,                        
                            OptixVisibilityMask( 255 ),
                            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                            SHADOW_RAY_TYPE,
                            RAY_TYPE_COUNT, 
                            SHADOW_RAY_TYPE,
                            u0, u1 );
                pixelColor += lightVisibility * optixLaunchParams.mLight.mPower * diffuseColor *
                              ( NdotL / ( lightDist * lightDist * numLightSamples ) );
            }
        }

        prd.mPixelNormal = Ns;
        prd.mPixelAlbedo = diffuseColor;
        prd.mPixelColor  = pixelColor;
    }

    extern "C" CUDA_KERNEL_DEFINITION void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */
    }

    extern "C" CUDA_KERNEL_DEFINITION void __anyhit__shadow()
    { /*! not going to be used */
    }

    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------

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
                        float3{ rayDir.x, rayDir.y, rayDir.z },
                        0.f,  
                        1e20f,
                        0.0f, 
                        OptixVisibilityMask( 255 ),
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                        RADIANCE_RAY_TYPE,            
                        RAY_TYPE_COUNT,               
                        RADIANCE_RAY_TYPE,            
                        u0, u1 );
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
