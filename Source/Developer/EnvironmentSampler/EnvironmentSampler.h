#pragma once

#include <memory>

#include "LaunchParams.h"

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Cuda/CudaBuffer.h"
#include "Cuda/MultiTensor.h"

#include "Developer/Scene/Components.h"
#include "Developer/Scene/Scene.h"

#include "Developer/Core/Optix/OptixAccelerationStructure.h"
#include "Developer/Core/Optix/OptixContext.h"
#include "Developer/Core/Optix/OptixModule.h"
#include "Developer/Core/Optix/OptixPipeline.h"
#include "Developer/Core/Optix/OptixShaderBindingTable.h"

#include "LaunchParams.h"

namespace LTSE::SensorModel::Dev
{

    using namespace LTSE::Core;
    using namespace LTSE::Cuda;
    using namespace LTSE::Graphics;

    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

        RaygenRecord()  = default;
        ~RaygenRecord() = default;
    };

    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

        MissRecord()  = default;
        ~MissRecord() = default;
    };

    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        TriangleMeshSBTData data{};

        HitgroupRecord()  = default;
        ~HitgroupRecord() = default;

        HitgroupRecord( TriangleMeshSBTData a_Data )
            : data{ a_Data }
        {
        }
    };

    class WorldSampler
    {
      public:
        WorldSampler( Ref<OptixDeviceContextObject> a_RayTracingContext );
        ~WorldSampler() = default;

        void Sample( math::mat4 a_SensorTransform, Ref<Scene> a_Scene, MultiTensor &a_Azimuths, MultiTensor &a_Elevations, MultiTensor &a_Intensities, MultiTensor &a_SamplePoints );

      private:
        void BuildShaderBindingTable( Ref<Scene> a_Scene );

      private:
        CUcontext cudaContext;
        CUstream stream;

        Ref<OptixDeviceContextObject> m_RayTracingContext = nullptr;
        Ref<OptixModuleObject> m_RayTracingModule         = nullptr;
        Ref<OptixPipelineObject> m_RayTracingPipeline     = nullptr;
        Ref<OptixShaderBindingTableObject> m_SBT          = nullptr;

        GPUMemory m_RaygenRecordsBuffer;
        GPUMemory m_MissRecordsBuffer;
        GPUMemory m_HitgroupRecordsBuffer;

        LaunchParams m_LaunchParams;
        GPUMemory m_LaunchParamsBuffer;
    };

} // namespace LTSE::SensorModel::Dev
