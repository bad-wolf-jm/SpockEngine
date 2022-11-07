using SpockEngine

namespace Test 
{

    struct SensorActor : sActorComponent
    {
        Ref<Scope>        mComputeScope = nullptr;
        Ref<WorldSampler> mWorldSampler = nullptr;
        Ref<EngineLoop>   mEngineLoop    = nullptr;
        Ref<Scene>        mWorld        = nullptr;

        sPointCloudVisualizer m_PointCloudVisualizer{};

        SensorControllerBehaviour( Ref<EngineLoop> aEngineLoop, Ref<Scene> aWorld )
            : mEngineLoop{ aEngineLoop }
            , mWorld{ aWorld }
        {
        }

        void OnCreate()
        {
            mComputeScope = New<Scope>( 512 * 1024 * 1024 );
            mWorldSampler = New<WorldSampler>( mWorld->GetRayTracingContext() );
        }

        void OnDestroy() {}

        void OnUpdate( Timestep ts )
        {
            if( !Has<sTransformMatrixComponent>() ) return;

            sRandomUniformInitializerComponent lInitializer{};
            lInitializer.mType = eScalarType::FLOAT32;

            std::vector<uint32_t> lDim1{ 2500, 2000 };

            OpNode lAzimuths    = MultiTensorValue( mComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );
            OpNode lElevations  = MultiTensorValue( mComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );
            OpNode lIntensities = MultiTensorValue( mComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );

            OpNode lRange = ConstantScalarValue( *mComputeScope, 25.0f );

            lAzimuths   = Multiply( mComputeScope, lAzimuths, lRange );
            lElevations = Multiply( mComputeScope, lElevations, lRange );
            mComputeScope.Run( { lAzimuths, lElevations, lIntensities } );

            sTensorShape lOutputShape( lIntensities.GetMultiTensor().Shape().mShape, sizeof( sHitRecord ) );
            MultiTensor  lHitRecords = MultiTensor( mComputeScope->mPool, lOutputShape );

            auto &lParticles = Get<sParticleSystemComponent>();

            mWorldSampler->Sample( Get<sTransformMatrixComponent>().Matrix, mWorld, lAzimuths.GetMultiTensor(),
                lElevations.GetMultiTensor(), lIntensities.GetMultiTensor(), lHitRecords );

            if( !( lParticles.Particles ) || lParticles.ParticleCount != lAzimuths.GetMultiTensor().SizeAs<float>() )
            {
                lParticles.ParticleCount = lAzimuths.GetMultiTensor().SizeAs<float>();
                lParticles.Particles = New<Buffer>( mEngineLoop->GetGraphicContext(), eBufferBindType::VERTEX_BUFFER, false, true, true,
                    true, lParticles.ParticleCount * sizeof( Particle ) );
            }

            GPUExternalMemory lPointCloudMappedBuffer( *( lParticles.Particles ), lParticles.ParticleCount * sizeof( Particle ) );
            m_PointCloudVisualizer.mInvertZAxis = false;
            m_PointCloudVisualizer.mResolution  = 0.2;
            m_PointCloudVisualizer.Visualize( Get<sTransformMatrixComponent>().Matrix, lHitRecords, lPointCloudMappedBuffer );
            lPointCloudMappedBuffer.Dispose();
        }
    };
}