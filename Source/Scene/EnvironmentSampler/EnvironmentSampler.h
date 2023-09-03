#pragma once

#include <memory>

#include "LaunchParams.h"

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Core/CUDA/Array/CudaBuffer.h"
#include "Core/CUDA/Array/MultiTensor.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "Core/Optix/OptixAccelerationStructure.h"
#include "Core/Optix/OptixContext.h"
#include "Core/Optix/OptixModule.h"
#include "Core/Optix/OptixPipeline.h"
#include "Core/Optix/OptixShaderBindingTable.h"

#include "LaunchParams.h"

namespace SE::SensorModel::Dev
{
    using namespace math;
    using namespace SE::Core;
    using namespace SE::Cuda;
    using namespace SE::Graphics;

    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) sRaygenRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char mHeader[OPTIX_SBT_RECORD_HEADER_SIZE];

        sRaygenRecord()  = default;
        ~sRaygenRecord() = default;
    };

    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) sMissRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char mHeader[OPTIX_SBT_RECORD_HEADER_SIZE];

        sMissRecord()  = default;
        ~sMissRecord() = default;
    };

    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) sHitgroupRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char mHeader[OPTIX_SBT_RECORD_HEADER_SIZE];
        TriangleMeshSBTData                          mData{};

        sHitgroupRecord()  = default;
        ~sHitgroupRecord() = default;

        sHitgroupRecord( TriangleMeshSBTData aData )
            : mData{ aData }
        {
        }
    };

    class WorldSampler
    {
      public:
        WorldSampler( ref_t<OptixDeviceContextObject> a_RayTracingContext );
        ~WorldSampler() = default;

        void Sample( math::mat4 a_SensorTransform, ref_t<Scene> a_Scene, MultiTensor &a_Azimuths, MultiTensor &a_Elevations,
                     MultiTensor &a_Intensities, MultiTensor &a_SamplePoints );

      private:
        void BuildShaderBindingTable( ref_t<Scene> a_Scene );

      private:
        CUcontext mCudaContext;
        CUstream  mCudaStream;

        ref_t<OptixDeviceContextObject>      mRayTracingContext  = nullptr;
        ref_t<OptixModuleObject>             mRayTracingModule   = nullptr;
        ref_t<OptixPipelineObject>           mRayTracingPipeline = nullptr;
        ref_t<OptixShaderBindingTableObject> mSBT                = nullptr;

        GPUMemory mRaygenRecordsBuffer;
        GPUMemory mMissRecordsBuffer;
        GPUMemory mHitgroupRecordsBuffer;

        LaunchParams mLaunchParams;
        GPUMemory    mLaunchParamsBuffer;
    };

} // namespace SE::SensorModel::Dev
