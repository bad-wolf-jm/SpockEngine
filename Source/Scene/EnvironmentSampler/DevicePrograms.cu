#include <optix_device.h>

#include "LaunchParams.h"

namespace SE::SensorModel::Dev
{

    static constexpr float PI      = 3.14159265359;
    static constexpr float RADIANS = PI / 180.0;
    static constexpr float HALF_PI = 1.57079632679;

    extern "C" __constant__ LaunchParams gOptixLaunchParams;

    enum
    {
        SURFACE_RAY_TYPE = 0,
        RAY_TYPE_COUNT
    };

    static __forceinline__ __device__ void *UnpackPointer( uint32_t i0, uint32_t i1 )
    {
        const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
        void          *ptr  = reinterpret_cast<void *>( uptr );
        return ptr;
    }

    static __forceinline__ __device__ void PackPointer( void *ptr, uint32_t &i0, uint32_t &i1 )
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
        i0                  = uptr >> 32;
        i1                  = uptr & 0x00000000ffffffff;
    }

    template <typename T>
    static __forceinline__ __device__ T *GetPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T *>( UnpackPointer( u0, u1 ) );
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        const uint32_t lRayID  = optixGetLaunchIndex().x;
        sHitRecord    &lPerRayData = *(sHitRecord *)GetPRD<sHitRecord>();

        lPerRayData.mRayID     = lRayID;
        lPerRayData.mAzimuth   = gOptixLaunchParams.mAzimuths[lRayID];
        lPerRayData.mElevation = gOptixLaunchParams.mElevations[lRayID];

        const TriangleMeshSBTData &lSbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();

        const int lPrimitiveID = optixGetPrimitiveIndex() + lSbtData.mIndexOffset;

        const math::vec3 &A =
            gOptixLaunchParams
                .mVertexBuffer[lSbtData.mVertexOffset + gOptixLaunchParams.mIndexBuffer[lPrimitiveID].x]
                .Position;
        const math::vec3 &B =
            gOptixLaunchParams
                .mVertexBuffer[lSbtData.mVertexOffset + gOptixLaunchParams.mIndexBuffer[lPrimitiveID].y]
                .Position;
        const math::vec3 &C =
            gOptixLaunchParams
                .mVertexBuffer[lSbtData.mVertexOffset + gOptixLaunchParams.mIndexBuffer[lPrimitiveID].z]
                .Position;

        const float u              = optixGetTriangleBarycentrics().x;
        const float v              = optixGetTriangleBarycentrics().y;

        math::vec3  lIntersection = A * ( 1.0f - u - v ) + B * u + C * v;

        lPerRayData.mDistance  = glm::length( lIntersection - gOptixLaunchParams.mSensorPosition );
        lPerRayData.mIntensity = gOptixLaunchParams.mIntensities[lRayID] / ( 1 + 4.0f * lPerRayData.mDistance * lPerRayData.mDistance );
    }

    extern "C" __global__ void __anyhit__radiance() {}

    extern "C" __global__ void __miss__radiance()
    {
        const uint32_t lRayID  = optixGetLaunchIndex().x;
        sHitRecord    &lPerRayData = *(sHitRecord *)GetPRD<sHitRecord>();

        lPerRayData.mRayID     = lRayID;
        lPerRayData.mAzimuth   = gOptixLaunchParams.mAzimuths[lRayID];
        lPerRayData.mElevation = gOptixLaunchParams.mElevations[lRayID];
        lPerRayData.mDistance  = 0.0f;
        lPerRayData.mIntensity = 0.0f;
    }

    extern "C" __global__ void __raygen__renderFrame()
    {
        const uint32_t lRayID = optixGetLaunchIndex().x;

        math::vec3 lSensorPosition = gOptixLaunchParams.mSensorPosition;

        sHitRecord lOutputSamplePoint{};

        uint32_t u0, u1;
        PackPointer( &lOutputSamplePoint, u0, u1 );

        float lPhi = HALF_PI - gOptixLaunchParams.mElevations[lRayID] * RADIANS;

        // With -z pointing inside the screen and x to the right,  3.0f * HALF_PI is inside. The azimuth
        // value moves to the right as it increases, but with the flipping of the x-axis due to the 3*HALF_PI
        // rotation, its sign is correct.
        float lTheta = 3.0f * HALF_PI + gOptixLaunchParams.mAzimuths[lRayID] * RADIANS;

        // -z points inside the screen
        math::vec3 lUnitDirectionVector;
        lUnitDirectionVector.x = glm::sin( lPhi ) * glm::cos( lTheta );
        lUnitDirectionVector.z = glm::sin( lPhi ) * glm::sin( lTheta );
        lUnitDirectionVector.y = glm::cos( lPhi );

        lUnitDirectionVector = gOptixLaunchParams.mSensorRotation * lUnitDirectionVector;

        optixTrace( gOptixLaunchParams.mTraversable, float3{ lSensorPosition.x, lSensorPosition.y, lSensorPosition.z },
            float3{ lUnitDirectionVector.x, lUnitDirectionVector.y, lUnitDirectionVector.z },
            0.f,   // tmin
            1e20f, // tmax
            0.0f,  // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
            SURFACE_RAY_TYPE,              // SBT offset
            RAY_TYPE_COUNT,                // SBT stride
            SURFACE_RAY_TYPE,              // missSBTIndex
            u0, u1 );

        gOptixLaunchParams.mSamplePoints[lRayID] = lOutputSamplePoint;
    }

} // namespace SE::SensorModel::Dev
