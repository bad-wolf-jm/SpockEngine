#include "EnvironmentSampler.h"
#include "Core/Logging.h"
#include "Core/Profiling/BlockTimer.h"

namespace SE::SensorModel::Dev
{

    extern "C" char g_SensorModelEnvironmentSampler[];

    WorldSampler::WorldSampler( ref_t<OptixDeviceContextObject> aRayTracingContext )
        : mRayTracingContext{ aRayTracingContext }
    {
        mRayTracingModule = New<OptixModuleObject>( "gOptixLaunchParams", g_SensorModelEnvironmentSampler, mRayTracingContext );

        mRayTracingModule->CreateRayGenGroup( "__raygen__renderFrame" );
        mRayTracingModule->CreateMissGroup( "__miss__radiance" );
        mRayTracingModule->CreateHitGroup( "__closesthit__radiance", "__anyhit__radiance" );

        mRayTracingPipeline = mRayTracingModule->CreatePipeline();
        mLaunchParamsBuffer = GPUMemory::Create<LaunchParams>( 1 );
    }

    void WorldSampler::BuildShaderBindingTable( ref_t<Scene> a_Scene )
    {
        mSBT = New<OptixShaderBindingTableObject>();

        // build raygen records
        vec_t<sRaygenRecord> lRaygenRecords = mSBT->NewRecordType<sRaygenRecord>( mRayTracingModule->mRayGenProgramGroups );
        mRaygenRecordsBuffer                      = GPUMemory::Create( lRaygenRecords );
        mSBT->BindRayGenRecordTable( mRaygenRecordsBuffer.RawDevicePtr() );

        // build miss records
        vec_t<sMissRecord> lMissRecords = mSBT->NewRecordType<sMissRecord>( mRayTracingModule->mMissProgramGroups );
        mMissRecordsBuffer                    = GPUMemory::Create( lMissRecords );
        mSBT->BindMissRecordTable<sMissRecord>( mMissRecordsBuffer.RawDevicePtr(), mMissRecordsBuffer.SizeAs<sMissRecord>() );

        vec_t<sHitgroupRecord> lHitgroupRecords;
        a_Scene->ForEach<SE::Core::EntityComponentSystem::Components::sRayTracingTargetComponent,
                         SE::Core::EntityComponentSystem::Components::sStaticMeshComponent>(
            [&]( auto l_Entity, auto &l_Component, auto &aMeshComponent )
            {
                sHitgroupRecord rec     = mSBT->NewRecordType<sHitgroupRecord>( mRayTracingModule->mHitProgramGroups[0] );
                rec.mData.mVertexOffset = aMeshComponent.mVertexOffset;
                rec.mData.mIndexOffset  = aMeshComponent.mIndexOffset / 3;
                lHitgroupRecords.push_back( rec );
            } );
        mHitgroupRecordsBuffer = GPUMemory::Create( lHitgroupRecords );
        mSBT->BindHitRecordTable<sHitgroupRecord>( mHitgroupRecordsBuffer.RawDevicePtr(), mHitgroupRecordsBuffer.Size() );
    }

    void WorldSampler::Sample( mat4 a_SensorTransform, ref_t<Scene> a_Scene, MultiTensor &a_Azimuths, MultiTensor &a_Elevations,
                               MultiTensor &a_Intensities, MultiTensor &a_SamplePoints )
    {
        SE_PROFILE_FUNCTION();

        BuildShaderBindingTable( a_Scene );

        if( a_Scene->GetRayTracingRoot() )
        {
            mLaunchParams.mTraversable    = a_Scene->GetRayTracingRoot();
            mLaunchParams.mSensorPosition = Translation( a_SensorTransform );
            mLaunchParams.mSensorRotation = NormalMatrix( a_SensorTransform );
            mLaunchParams.mAzimuths       = a_Azimuths.DataAs<float>();
            mLaunchParams.mElevations     = a_Elevations.DataAs<float>();
            mLaunchParams.mIntensities    = a_Intensities.DataAs<float>();
            mLaunchParams.mSamplePoints   = a_SamplePoints.DataAs<sHitRecord>();

            // mLaunchParams.mIndexBuffer  = a_Scene->mIndexBuffer->DataAs<uvec3>();
            // mLaunchParams.mVertexBuffer = a_Scene->mTransformedVertexBuffer->DataAs<VertexData>();

            mLaunchParamsBuffer.Upload( mLaunchParams );
            mRayTracingPipeline->Launch( 0, mLaunchParamsBuffer.RawDevicePtr(), mLaunchParamsBuffer.Size(), mSBT,
                                         uvec3{ a_Azimuths.SizeAs<float>(), 1, 1 } );
            SyncDevice();
        }
    }

} // namespace SE::SensorModel::Dev
