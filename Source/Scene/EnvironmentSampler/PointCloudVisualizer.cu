#include "PointCloudVisualizer.h"

#include <iostream>
#include <stdio.h>

#include "Core/Math/Types.h"

#include "LaunchParams.h"

#define THREADS_PER_BLOCK 512

namespace SE::SensorModel::Dev
{

    static constexpr float PI      = 3.14159265359;
    static constexpr float RADIANS = PI / 180.0;
    static constexpr float HALF_PI = 1.57079632679;

    __device__ math::vec3 hsv2rgb( math::vec3 c )
    {
        float r, g, b;
        float h, s, v;

        h = c.x;
        s = c.y;
        v = c.z;

        float f  = h / 60.0f;
        float hi = floorf( f );
        f        = f - hi;
        float p  = v * ( 1 - s );
        float q  = v * ( 1 - s * f );
        float t  = v * ( 1 - s * ( 1 - f ) );

        if( hi == 0.0f || hi == 6.0f )
        {
            r = v;
            g = t;
            b = p;
        }
        else if( hi == 1.0f )
        {
            r = q;
            g = v;
            b = p;
        }
        else if( hi == 2.0f )
        {
            r = p;
            g = v;
            b = t;
        }
        else if( hi == 3.0f )
        {
            r = p;
            g = q;
            b = v;
        }
        else if( hi == 4.0f )
        {
            r = t;
            g = p;
            b = v;
        }
        else
        {
            r = v;
            g = p;
            b = q;
        }

        return math::vec3( r, g, b );
    }

    __device__ math::vec3 IntensityToRGB( float I, float a_Offset )
    {
        I = a_Offset - I * a_Offset;
        return hsv2rgb( math::vec3( I, 1.0f, 1.0f ) );
    }

    extern "C" __device__ sLidarCartesianSamplePoint LidarDataToCartesian( math::mat3 a_PointCloudRotation,
                                                                           sHitRecord a_LidarReturnPoints, bool a_InvertZAxis )
    {
        sLidarCartesianSamplePoint l_CartesianSample{};

        float l_Phi = HALF_PI - a_LidarReturnPoints.mElevation * RADIANS;

        // With -z pointing inside the screen and x to the right,  3.0f * HALF_PI is inside. The azimuth
        // value moves to the right as it increases, but with the flipping of the x-axis due to the 3*HALF_PI
        // rotation, its sign is correct.
        float l_Theta = 3.0f * HALF_PI + a_LidarReturnPoints.mAzimuth * RADIANS;

        l_CartesianSample.mFlashID = 0; // a_LidarReturnPoints.FlashID;
        l_CartesianSample.mType    = 0; // a_LidarReturnPoints.Type;

        l_CartesianSample.mDirection.x = ::sin( l_Phi ) * ::cos( l_Theta );
        l_CartesianSample.mDirection.z = ::sin( l_Phi ) * ::sin( l_Theta );
        l_CartesianSample.mDirection.y = ::cos( l_Phi );

        l_CartesianSample.mDirection = a_PointCloudRotation * l_CartesianSample.mDirection;
        l_CartesianSample.mDistance  = a_LidarReturnPoints.mDistance;
        l_CartesianSample.mIntensity = a_LidarReturnPoints.mIntensity;

        if( a_InvertZAxis )
        {
            l_CartesianSample.mDirection.z = -l_CartesianSample.mDirection.z;
        }

        return l_CartesianSample;
    }

    extern "C" __global__ void __kernel__FillParticleBuffer( math::mat3 a_PointCloudRotation, math::vec3 a_PointCloudOrigin,
                                                             MultiTensor a_LidarReturnPoints, GPUExternalMemory a_LidarParticle,
                                                             bool a_LogScale, bool a_HighlightFOV, bool a_InvertZAxis,
                                                             float aResolution )
    {
        int l_InputArrayIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if( l_InputArrayIdx >= a_LidarReturnPoints.SizeAs<sHitRecord>() ) return;

        sHitRecord *lHitRecords    = a_LidarReturnPoints.DataAs<sHitRecord>();
        Particle   *lLidarParticle = a_LidarParticle.DataAs<Particle>();

        sLidarCartesianSamplePoint l_CartesianPoint =
            LidarDataToCartesian( a_PointCloudRotation, lHitRecords[l_InputArrayIdx], a_InvertZAxis );

        float I = lHitRecords[l_InputArrayIdx].mIntensity;

        if( a_LogScale )
        {
            if( I <= 1e-9 )
            {
                I = 0.0f;
            }
            else
            {
                I = logf( I ) / 9.0f + 1;
            }
        }

        math::vec3 l_Color = IntensityToRGB( I, 240.0f );

        lLidarParticle[l_InputArrayIdx].PositionAndSize =
            math::vec4( a_PointCloudOrigin + l_CartesianPoint.mDirection * l_CartesianPoint.mDistance,
                        glm::tan( aResolution / 2.0f ) * l_CartesianPoint.mDistance * 2.0f );
        lLidarParticle[l_InputArrayIdx].Color = math::vec4( l_Color, 0.95 );
    }

    void sPointCloudVisualizer::Visualize( math::mat4 a_PointCloudTransform, MultiTensor &a_LidarReturnPoints,
                                           GPUExternalMemory &a_Particles )
    {
        int l_BlockCount = ( a_LidarReturnPoints.SizeAs<sHitRecord>() / THREADS_PER_BLOCK ) + 1;

        dim3 l_GridDim( static_cast<int>( l_BlockCount ), 1, 1 );
        dim3 l_BlockDim( THREADS_PER_BLOCK );

        a_Particles.Zero();
        __kernel__FillParticleBuffer<<<l_GridDim, l_BlockDim>>>(
            math::NormalMatrix( a_PointCloudTransform ), math::Translation( a_PointCloudTransform ), a_LidarReturnPoints, a_Particles,
            mLogScale, mHighlightFlashFOV, mInvertZAxis, mResolution * RADIANS );
        cudaDeviceSynchronize();
    }

} // namespace SE::SensorModel::Dev
