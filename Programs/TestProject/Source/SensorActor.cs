using SpockEngine;

namespace Test 
{
    public class SensorActor : ActorComponent
    {
        // private Scope        mComputeScope;
        // private WorldSampler mWorldSampler;
        // Ref<EngineLoop>   mEngineLoop = nullptr;
        // Ref<Scene>        mWorld      = nullptr;

        // sPointCloudVisualizer mPointCloudVisualizer{};

        SensorActor( ) : base() {}

        override public void BeginScenario()
        {
            // mComputeScope = new Scope( 512 * 1024 * 1024 );
            // mWorldSampler = new WorldSampler( GetRayTracingContext() );
        }

        override public void EndScenario() {}

        override public void Tick( float ts )
        {
            // if( !Has<sTransformMatrixComponent>() ) return;

            // sRandomUniformInitializerComponent lInitializer{};
            // lInitializer.mType = eScalarType::FLOAT32;

            // std::vector<uint32_t> lDim1{ 2500, 2000 };

            // OpNode lAzimuths    = MultiTensorValue( mComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );
            // OpNode lElevations  = MultiTensorValue( mComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );
            // OpNode lIntensities = MultiTensorValue( mComputeScope, lInitializer, sTensorShape( { lDim1 }, sizeof( float ) ) );

            // OpNode lRange = ConstantScalarValue( mComputeScope, 25.0f );

            // lAzimuths   = Multiply( mComputeScope, lAzimuths, lRange );
            // lElevations = Multiply( mComputeScope, lElevations, lRange );
            // mComputeScope.Run( { lAzimuths, lElevations, lIntensities } );

            // sTensorShape lOutputShape( lIntensities.GetMultiTensor().Shape().mShape, sizeof( sHitRecord ) );
            // MultiTensor  lHitRecords = MultiTensor( mComputeScope->mPool, lOutputShape );

            // auto &lParticles = Get<sParticleSystemComponent>();

            // mWorldSampler->Sample( Get<sTransformMatrixComponent>().Matrix, mWorld, lAzimuths.GetMultiTensor(),
            //     lElevations.GetMultiTensor(), lIntensities.GetMultiTensor(), lHitRecords );

            // if( !( lParticles.Particles ) || lParticles.ParticleCount != lAzimuths.GetMultiTensor().SizeAs<float>() )
            // {
            //     lParticles.ParticleCount = lAzimuths.GetMultiTensor().SizeAs<float>();
            //     lParticles.Particles = New<Buffer>( mEngineLoop->GetGraphicContext(), eBufferType::VERTEX_BUFFER, false, true, true,
            //         true, lParticles.ParticleCount * sizeof( Particle ) );
            // }

            // GPUExternalMemory lPointCloudMappedBuffer( *( lParticles.Particles ), lParticles.ParticleCount * sizeof( Particle ) );
            // mPointCloudVisualizer.mInvertZAxis = false;
            // mPointCloudVisualizer.mResolution  = 0.2;
            // mPointCloudVisualizer.Visualize( Get<sTransformMatrixComponent>().Matrix, lHitRecords, lPointCloudMappedBuffer );
            // lPointCloudMappedBuffer.Dispose();
        }
    };
}