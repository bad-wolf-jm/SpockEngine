#include "EnvironmentSampler.h"
#include "Core/Logging.h"
#include "Core/Profiling/BlockTimer.h"

namespace LTSE::SensorModel::Dev
{

    extern "C" char g_SensorModelEnvironmentSampler[];

    WorldSampler::WorldSampler( Ref<OptixDeviceContextObject> a_RayTracingContext )
        : m_RayTracingContext{ a_RayTracingContext }
    {
        m_RayTracingModule =
            New<OptixModuleObject>( "gOptixLaunchParams", g_SensorModelEnvironmentSampler, m_RayTracingContext );

        m_RayTracingModule->CreateRayGenGroup( "__raygen__renderFrame" );
        m_RayTracingModule->CreateMissGroup( "__miss__radiance" );
        m_RayTracingModule->CreateHitGroup( "__closesthit__radiance", "__anyhit__radiance" );

        m_RayTracingPipeline = m_RayTracingModule->CreatePipeline();
        m_LaunchParamsBuffer = GPUMemory::Create<LaunchParams>( 1 );
    }

    void WorldSampler::BuildShaderBindingTable( Ref<Scene> a_Scene )
    {
        m_SBT = New<OptixShaderBindingTableObject>();

        // build raygen records
        std::vector<RaygenRecord> raygenRecords =
            m_SBT->NewRecordType<RaygenRecord>( m_RayTracingModule->m_RayGenProgramGroups );
        m_RaygenRecordsBuffer = GPUMemory::Create( raygenRecords );
        m_SBT->BindRayGenRecordTable( m_RaygenRecordsBuffer.RawDevicePtr() );

        // build miss records
        std::vector<MissRecord> missRecords = m_SBT->NewRecordType<MissRecord>( m_RayTracingModule->m_MissProgramGroups );
        m_MissRecordsBuffer                 = GPUMemory::Create( missRecords );
        m_SBT->BindMissRecordTable<MissRecord>(
            m_MissRecordsBuffer.RawDevicePtr(), m_MissRecordsBuffer.SizeAs<MissRecord>() );

        std::vector<HitgroupRecord> hitgroupRecords;
        a_Scene->ForEach<LTSE::Core::EntityComponentSystem::Components::sRayTracingTargetComponent,
            LTSE::Core::EntityComponentSystem::Components::sStaticMeshComponent>(
            [&]( auto l_Entity, auto& l_Component, auto& aMeshComponent )
            {
                HitgroupRecord rec     = m_SBT->NewRecordType<HitgroupRecord>( m_RayTracingModule->m_HitProgramGroups[0] );
                rec.data.mVertexOffset = aMeshComponent.mVertexOffset;
                rec.data.mIndexOffset  = aMeshComponent.mIndexOffset / 3;
                hitgroupRecords.push_back( rec );
            } );
        m_HitgroupRecordsBuffer = GPUMemory::Create( hitgroupRecords );
        m_SBT->BindHitRecordTable<HitgroupRecord>( m_HitgroupRecordsBuffer.RawDevicePtr(), m_HitgroupRecordsBuffer.Size() );
    }

    void WorldSampler::Sample( math::mat4 a_SensorTransform, Ref<Scene> a_Scene, MultiTensor& a_Azimuths,
        MultiTensor& a_Elevations, MultiTensor& a_Intensities, MultiTensor& a_SamplePoints )
    {
        LTSE_PROFILE_FUNCTION();

        BuildShaderBindingTable( a_Scene );

        if( a_Scene->GetRayTracingRoot() )
        {
            m_LaunchParams.traversable    = a_Scene->GetRayTracingRoot();
            m_LaunchParams.SensorPosition = math::Translation( a_SensorTransform );
            m_LaunchParams.SensorRotation = math::NormalMatrix( a_SensorTransform );
            m_LaunchParams.Azimuths       = a_Azimuths.DataAs<float>();
            m_LaunchParams.Elevations     = a_Elevations.DataAs<float>();
            m_LaunchParams.Intensities    = a_Intensities.DataAs<float>();
            m_LaunchParams.SamplePoints   = a_SamplePoints.DataAs<HitRecord>();

            LTSE::Cuda::GPUExternalMemory lTransformedVertexBuffer(
                *a_Scene->mTransformedVertexBuffer, a_Scene->mTransformedVertexBuffer->SizeAs<uint8_t>() );
            LTSE::Cuda::GPUExternalMemory lIndexBuffer( *a_Scene->mIndexBuffer, a_Scene->mVertexBuffer->SizeAs<uint8_t>() );
            m_LaunchParams.mIndexBuffer  = lIndexBuffer.DataAs<math::uvec3>();
            m_LaunchParams.mVertexBuffer = lTransformedVertexBuffer.DataAs<VertexData>();

            m_LaunchParamsBuffer.Upload( m_LaunchParams );
            m_RayTracingPipeline->Launch( 0, m_LaunchParamsBuffer.RawDevicePtr(), m_LaunchParamsBuffer.Size(), m_SBT,
                math::uvec3{ a_Azimuths.SizeAs<float>(), 1, 1 } );

            CUDA_SYNC_CHECK();

            lTransformedVertexBuffer.Dispose();
            lIndexBuffer.Dispose();
        }
    }

} // namespace LTSE::SensorModel::Dev
