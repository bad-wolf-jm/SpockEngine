#include <optix_device.h>

#include "LaunchParams.h"

namespace LTSE::SensorModel::Dev
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
        void *ptr           = reinterpret_cast<void *>( uptr );
        return ptr;
    }

    static __forceinline__ __device__ void PackPointer( void *ptr, uint32_t &i0, uint32_t &i1 )
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
        i0                  = uptr >> 32;
        i1                  = uptr & 0x00000000ffffffff;
    }

    template <typename T> static __forceinline__ __device__ T *GetPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T *>( UnpackPointer( u0, u1 ) );
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        const uint32_t ix = optixGetLaunchIndex().x;
        HitRecord &prd    = *(HitRecord *)GetPRD<HitRecord>();

        prd.RayID     = ix;
        prd.Azimuth   = gOptixLaunchParams.Azimuths[ix];
        prd.Elevation = gOptixLaunchParams.Elevations[ix];

        const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();

        const int l_PrimitiveID = optixGetPrimitiveIndex();

        const math::vec3 &A = gOptixLaunchParams.mVertexBuffer[sbtData.mVertexOffset + gOptixLaunchParams.mIndexBuffer[sbtData.mIndexOffset + l_PrimitiveID].x].Position;
        const math::vec3 &B = gOptixLaunchParams.mVertexBuffer[sbtData.mVertexOffset + gOptixLaunchParams.mIndexBuffer[sbtData.mIndexOffset + l_PrimitiveID].y].Position;
        const math::vec3 &C = gOptixLaunchParams.mVertexBuffer[sbtData.mVertexOffset + gOptixLaunchParams.mIndexBuffer[sbtData.mIndexOffset + l_PrimitiveID].z].Position;

        const float u             = optixGetTriangleBarycentrics().x;
        const float v             = optixGetTriangleBarycentrics().y;
        math::vec3 l_Intersection = A * ( 1.0f - u - v ) + B * u + C * v;

        prd.Distance  = glm::length( l_Intersection - gOptixLaunchParams.SensorPosition );
        prd.Intensity = gOptixLaunchParams.Intensities[ix] / ( 1 + 4.0f * prd.Distance * prd.Distance );
    }

    extern "C" __global__ void __anyhit__radiance() {}

    extern "C" __global__ void __miss__radiance()
    {
        const uint32_t ix = optixGetLaunchIndex().x;
        HitRecord &prd    = *(HitRecord *)GetPRD<HitRecord>();

        prd.RayID     = ix;
        prd.Azimuth   = gOptixLaunchParams.Azimuths[ix];
        prd.Elevation = gOptixLaunchParams.Elevations[ix];
        prd.Distance  = 0.0f;
        prd.Intensity = 0.0f;
    }

    extern "C" __global__ void __raygen__renderFrame()
    {
        const uint32_t ix = optixGetLaunchIndex().x;

        math::vec3 l_SensorPosition = gOptixLaunchParams.SensorPosition;

        HitRecord l_OutputSamplePoint{};

        uint32_t u0, u1;
        PackPointer( &l_OutputSamplePoint, u0, u1 );

        float l_Phi = HALF_PI - gOptixLaunchParams.Elevations[ix] * RADIANS;

        // With -z pointing inside the screen and x to the right,  3.0f * HALF_PI is inside. The azimuth
        // value moves to the right as it increases, but with the flipping of the x-axis due to the 3*HALF_PI
        // rotation, its sign is correct.
        float l_Theta = 3.0f * HALF_PI + gOptixLaunchParams.Azimuths[ix] * RADIANS;

        // -z points inside the screen
        math::vec3 l_UnitDirectionVector;
        l_UnitDirectionVector.x = glm::sin( l_Phi ) * glm::cos( l_Theta );
        l_UnitDirectionVector.z = glm::sin( l_Phi ) * glm::sin( l_Theta );
        l_UnitDirectionVector.y = glm::cos( l_Phi );

        l_UnitDirectionVector = gOptixLaunchParams.SensorRotation * l_UnitDirectionVector;

        optixTrace( gOptixLaunchParams.traversable, float3{ l_SensorPosition.x, l_SensorPosition.y, l_SensorPosition.z },
                    float3{ l_UnitDirectionVector.x, l_UnitDirectionVector.y, l_UnitDirectionVector.z },
                    0.f,   // tmin
                    1e20f, // tmax
                    0.0f,  // rayTime
                    OptixVisibilityMask( 255 ),
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
                    SURFACE_RAY_TYPE,              // SBT offset
                    RAY_TYPE_COUNT,                // SBT stride
                    SURFACE_RAY_TYPE,              // missSBTIndex
                    u0, u1 );

        gOptixLaunchParams.SamplePoints[ix] = l_OutputSamplePoint;
    }

} // namespace LTSE::SensorModel::Dev
