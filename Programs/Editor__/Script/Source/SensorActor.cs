using SpockEngine

namespace Test 
{

    struct SensorActor : sActorComponent
    {
        Ref<Scope>        m_ComputeScope = nullptr;
        Ref<WorldSampler> m_WorldSampler = nullptr;
        Ref<EngineLoop>   mEngineLoop    = nullptr;
        Ref<Scene>        m_World        = nullptr;

        sPointCloudVisualizer m_PointCloudVisualizer{};

        SensorControllerBehaviour( Ref<EngineLoop> aEngineLoop, Ref<Scene> aWorld )
            : mEngineLoop{ aEngineLoop }
            , m_World{ aWorld }
        {
        }

        void OnCreate()
        {
            m_ComputeScope = New<Scope>( 512 * 1024 * 1024 );
            m_WorldSampler = New<WorldSampler>( m_World->GetRayTracingContext() );
        }

        void OnDestroy() {}

        void OnUpdate( Timestep ts )
        {
            sRandomUniformInitializerComponent lInitializer{};
            lInitializer.mType = eScalarType::FLOAT32;

            std::vector<uint32_t> lDim1{ 2500, 2000 };

            auto lAzimuths    = MultiTensorValue( *m_ComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );
            auto lElevations  = MultiTensorValue( *m_ComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );
            auto lIntensities = MultiTensorValue( *m_ComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );

            auto lRange = ConstantScalarValue( *m_ComputeScope, 25.0f );

            lAzimuths   = Multiply( *m_ComputeScope, lAzimuths, lRange );
            lElevations = Multiply( *m_ComputeScope, lElevations, lRange );
            m_ComputeScope->Run( { lAzimuths, lElevations, lIntensities } );

            sTensorShape lOutputShape( lIntensities.Get<sMultiTensorComponent>().mValue.Shape().mShape, sizeof( sHitRecord ) );
            MultiTensor  lHitRecords = MultiTensor( m_ComputeScope->mPool, lOutputShape );
            if( !Has<sTransformMatrixComponent>() ) return;

            auto &lParticles = Get<sParticleSystemComponent>();

            m_WorldSampler->Sample( Get<sTransformMatrixComponent>().Matrix, m_World, lAzimuths.Get<sMultiTensorComponent>().mValue,
                lElevations.Get<sMultiTensorComponent>().mValue, lIntensities.Get<sMultiTensorComponent>().mValue, lHitRecords );

            if( !( lParticles.Particles ) || lParticles.ParticleCount != lAzimuths.Get<sMultiTensorComponent>().mValue.SizeAs<float>() )
            {
                lParticles.ParticleCount = lAzimuths.Get<sMultiTensorComponent>().mValue.SizeAs<float>();
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