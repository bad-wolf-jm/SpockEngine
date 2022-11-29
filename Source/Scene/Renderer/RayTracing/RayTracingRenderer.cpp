#include "Core/Cuda/CudaAssert.h"

#include "LaunchParams.h"
#include "RayTracingRenderer.h"

// #include <optix_function_table_definition.h>

namespace SE::Core
{
    using namespace SE::Cuda;

    extern "C" char embedded_ptx_code[];

    /*! SBT record for a raygen program */
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void *data;
    };

    /*! SBT record for a miss program */
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void *data;
    };

    /*! SBT record for a hitgroup program */
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
    {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        sTriangleMeshSBTData                         data;
    };

    /*! constructor - performs all setup, including initializing
      optix, creates mOptixModule->mOptixObject, pipeline, programs, SBT, etc. */
    RayTracingRenderer::RayTracingRenderer()

    {
        SE::Graphics::OptixDeviceContextObject::Initialize();

        std::cout << "#osc: creating optix context ..." << std::endl;
        mOptixContext      = New<OptixDeviceContextObject>();
        const int deviceID = 0;
        CUDA_CHECK( SetDevice( deviceID ) );
        CUDA_CHECK( StreamCreate( &stream ) );

        std::cout << "#osc: setting up module and pipeline ..." << std::endl;
        mOptixModule = New<OptixModuleObject>( "optixLaunchParams", embedded_ptx_code, mOptixContext );
        mOptixModule->CreateRayGenGroup( "__raygen__renderFrame" );
        mOptixModule->CreateMissGroup( "__miss__radiance" );
        mOptixModule->CreateHitGroup( "__closesthit__radiance", "__anyhit__radiance" );
        mOptixModule->CreateMissGroup( "__miss__shadow" );
        mOptixModule->CreateHitGroup( "__closesthit__shadow", "__anyhit__shadow" );
        mOptixPipeline = mOptixModule->CreatePipeline();

        launchParamsBuffer = GPUMemory( sizeof( launchParams ) ); //.alloc( sizeof( launchParams ) );
        std::cout << "#osc: context, mOptixModule->mOptixObject, pipeline, etc, all set up ..." << std::endl;
    }


    static void context_log_cb( unsigned int level, const char *tag, const char *message, void * )
    {
        fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
    }

    void WorldSampler::BuildShaderBindingTable()
    {
        mShaderBindingTable = New<OptixShaderBindingTableObject>();

        std::vector<RaygenRecord> lRaygenRecords =
            mShaderBindingTable->NewRecordType<RaygenRecord>( mOptixModule->mRayGenProgramGroups );
        raygenRecordsBuffer = GPUMemory::Create( lRaygenRecords );
        mShaderBindingTable->BindRayGenRecordTable( raygenRecordsBuffer.RawDevicePtr() );

        std::vector<MissRecord> lMissRecords = mShaderBindingTable->NewRecordType<MissRecord>( mOptixModule->mMissProgramGroups );
        missRecordsBuffer                   = GPUMemory::Create( lMissRecords );
        mShaderBindingTable->BindMissRecordTable<MissRecord>( missRecordsBuffer.RawDevicePtr(),
                                                              missRecordsBuffer.SizeAs<MissRecord>() );

        std::vector<sHitgroupRecord> lHitgroupRecords;
        a_Scene->ForEach<SE::Core::EntityComponentSystem::Components::sRayTracingTargetComponent,
                         SE::Core::EntityComponentSystem::Components::sStaticMeshComponent>(
            [&]( auto lEntity, auto &lComponent, auto &aMeshComponent )
            {
                for( int lRayTypeID = 0; lRayTypeID < RAY_TYPE_COUNT; lRayTypeID++ )
                {
                    HitgroupRecord rec =
                        mShaderBindingTable->NewRecordType<HitgroupRecord>( mOptixModule->mHitProgramGroups[lRayTypeID] );
                    rec.data.mColor = mesh->mDiffuse;
                    if( mesh->mDiffuseTextureID >= 0 && mesh->mDiffuseTextureID < textureObjects.size() )
                    {
                        rec.data.mHasTexture = true;
                        rec.data.mTexture    = textureObjects[mesh->mDiffuseTextureID];
                    }
                    else
                    {
                        rec.data.mHasTexture = false;
                    }
                    rec.data.mIndex    = mIndices[meshID].DataAs<math::ivec3>();
                    rec.data.mVertex   = mVertices[meshID].DataAs<math::vec3>();
                    rec.data.mNormal   = mNormals[meshID].DataAs<math::vec3>();
                    rec.data.mTexCoord = mTexCoords[meshID].DataAs<math::vec2>();
                    hitgroupRecords.push_back( rec );
                }
            } );
        mHitgroupRecordsBuffer = GPUMemory::Create( lHitgroupRecords );
        mSBT->BindHitRecordTable<sHitgroupRecord>( mHitgroupRecordsBuffer.RawDevicePtr(), mHitgroupRecordsBuffer.Size() );
    }

    void RayTracingRenderer::Update( Ref<Scene> aWorld )
    {
        launchParams.mSceneRoot = aWorld->GetRayTracingRoot();

        BuildShaderBindingTable();
    }

    /*! render one frame */
    void RayTracingRenderer::Render()
    {
        if( launchParams.mFrame.mSize.x == 0 ) return;

        if( !accumulate ) launchParams.mFrame.mFrameID = 0;
        launchParamsBuffer.Upload( launchParams );
        launchParams.mFrame.mFrameID++;

        launchParams.mNumLightSamples = 1;
        launchParams.mNumPixelSamples = 1;

        mOptixPipeline->Launch( stream, launchParamsBuffer.RawDevicePtr(), launchParamsBuffer.SizeAs<uint8_t>(), mShaderBindingTable,
                                math::uvec3{ launchParams.mFrame.mSize.x, launchParams.mFrame.mSize.y, 1 } );

        denoiserIntensity.Resize( sizeof( float ) );

        OptixDenoiserParams denoiserParams;
        denoiserParams.denoiseAlpha = 1; // OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;

#if OPTIX_VERSION >= 70300
        if( denoiserIntensity.SizeAs<uint8_t>() != sizeof( float ) )
        {
            denoiserIntensity.Resize( sizeof( float ) );
        };
#endif

        denoiserParams.hdrIntensity = denoiserIntensity.RawDevicePtr();
        if( accumulate )
            denoiserParams.blendFactor = 1.f / ( launchParams.mFrame.mFrameID );
        else
            denoiserParams.blendFactor = 0.0f;

        OptixImage2D inputLayer[3];
        inputLayer[0].data               = fbColor.RawDevicePtr();
        inputLayer[0].width              = launchParams.mFrame.mSize.x;
        inputLayer[0].height             = launchParams.mFrame.mSize.y;
        inputLayer[0].rowStrideInBytes   = launchParams.mFrame.mSize.x * sizeof( float4 );
        inputLayer[0].pixelStrideInBytes = sizeof( float4 );
        inputLayer[0].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        inputLayer[2].data               = fbNormal.RawDevicePtr();
        inputLayer[2].width              = launchParams.mFrame.mSize.x;
        inputLayer[2].height             = launchParams.mFrame.mSize.y;
        inputLayer[2].rowStrideInBytes   = launchParams.mFrame.mSize.x * sizeof( float4 );
        inputLayer[2].pixelStrideInBytes = sizeof( float4 );
        inputLayer[2].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        inputLayer[1].data               = fbAlbedo.RawDevicePtr();
        inputLayer[1].width              = launchParams.mFrame.mSize.x;
        inputLayer[1].height             = launchParams.mFrame.mSize.y;
        inputLayer[1].rowStrideInBytes   = launchParams.mFrame.mSize.x * sizeof( float4 );
        inputLayer[1].pixelStrideInBytes = sizeof( float4 );
        inputLayer[1].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        OptixImage2D outputLayer;
        outputLayer.data               = denoisedBuffer.RawDevicePtr();
        outputLayer.width              = launchParams.mFrame.mSize.x;
        outputLayer.height             = launchParams.mFrame.mSize.y;
        outputLayer.rowStrideInBytes   = launchParams.mFrame.mSize.x * sizeof( float4 );
        outputLayer.pixelStrideInBytes = sizeof( float4 );
        outputLayer.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        // -------------------------------------------------------
        if( denoiserOn )
        {
            OPTIX_CHECK( optixDenoiserComputeIntensity( denoiser, 0, &inputLayer[0], (CUdeviceptr)denoiserIntensity.RawDevicePtr(),
                                                        (CUdeviceptr)denoiserScratch.RawDevicePtr(),
                                                        denoiserScratch.SizeAs<uint8_t>() ) );

#if OPTIX_VERSION >= 70300
            OptixDenoiserGuideLayer denoiserGuideLayer = {};
            denoiserGuideLayer.albedo                  = inputLayer[1];
            denoiserGuideLayer.normal                  = inputLayer[2];

            OptixDenoiserLayer denoiserLayer = {};
            denoiserLayer.input              = inputLayer[0];
            denoiserLayer.output             = outputLayer;

            OPTIX_CHECK( optixDenoiserInvoke( denoiser, 0, &denoiserParams, denoiserState.RawDevicePtr(),
                                              denoiserState.SizeAs<uint8_t>(), &denoiserGuideLayer, &denoiserLayer, 1, 0, 0,
                                              denoiserScratch.RawDevicePtr(), denoiserScratch.SizeAs<uint8_t>() ) );
#else
            OPTIX_CHECK( optixDenoiserInvoke( denoiser, 0, &denoiserParams, denoiserState.RawDevicePtr(),
                                              denoiserState.SizeAs<uint8_t>(), &inputLayer[0], 2, 0, 0, &outputLayer,
                                              denoiserScratch.RawDevicePtr(), denoiserScratch.SizeAs<uint8_t>() ) );
#endif
        }
        else
        {
            cudaMemcpy( (void *)outputLayer.data, (void *)inputLayer[0].data,
                        outputLayer.width * outputLayer.height * sizeof( float4 ), cudaMemcpyDeviceToDevice );
        }

        computeFinalPixelColors();

        CUDA_SYNC_CHECK();
    }

    /*! set camera to render with */
    void RayTracingRenderer::setCamera( const Camera &camera )
    {
        lastSetCamera = camera;
        // reset accumulation
        launchParams.mFrame.mFrameID    = 0;
        launchParams.mCamera.mPosition  = camera.from;
        launchParams.mCamera.mDirection = normalize( camera.at - camera.from );
        const float cosFovy             = 0.66f;
        const float aspect              = float( launchParams.mFrame.mSize.x ) / float( launchParams.mFrame.mSize.y );
        launchParams.mCamera.mHorizontal =
            cosFovy * aspect * math::normalize( math::cross( launchParams.mCamera.mDirection, camera.up ) );
        launchParams.mCamera.mVertical =
            cosFovy * normalize( cross( launchParams.mCamera.mHorizontal, launchParams.mCamera.mDirection ) );
    }

    /*! resize frame buffer to given resolution */
    void RayTracingRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        if( denoiser )
        {
            OPTIX_CHECK( optixDenoiserDestroy( denoiser ) );
        };

        // ------------------------------------------------------------------
        // create the denoiser:
        OptixDenoiserOptions denoiserOptions = {};
#if OPTIX_VERSION >= 70300
        OPTIX_CHECK( optixDenoiserCreate( mOptixContext->mOptixObject, OPTIX_DENOISER_MODEL_KIND_LDR, &denoiserOptions, &denoiser ) );
#else
        denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;
#    if OPTIX_VERSION < 70100
        // these only exist in 7.0, not 7.1
        denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#    endif

        OPTIX_CHECK( optixDenoiserCreate( mOptixContext->mOptixObject, &denoiserOptions, &denoiser ) );
        OPTIX_CHECK( optixDenoiserSetModel( denoiser, OPTIX_DENOISER_MODEL_KIND_LDR, NULL, 0 ) );
#endif

        // .. then compute and allocate memory resources for the denoiser
        OptixDenoiserSizes denoiserReturnSizes;
        OPTIX_CHECK( optixDenoiserComputeMemoryResources( denoiser, aOutputWidth, aOutputHeight, &denoiserReturnSizes ) );

#if OPTIX_VERSION < 70100
        denoiserScratch.Resize( denoiserReturnSizes.recommendedScratchSizeInBytes );
#else
        denoiserScratch.Resize(
            std::max( denoiserReturnSizes.withOverlapScratchSizeInBytes, denoiserReturnSizes.withoutOverlapScratchSizeInBytes ) );
#endif
        denoiserState.Resize( denoiserReturnSizes.stateSizeInBytes );

        // ------------------------------------------------------------------
        // resize our cuda frame buffer

        denoisedBuffer.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        fbColor.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        fbNormal.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        fbAlbedo.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        finalColorBuffer.Resize( aOutputWidth * aOutputHeight * sizeof( uint32_t ) );

        // update the launch parameters that we'll pass to the optix
        // launch:
        launchParams.mFrame.mSize         = math::ivec2{ aOutputWidth, aOutputHeight };
        launchParams.mFrame.mColorBuffer  = (math::vec4 *)fbColor.RawDevicePtr();
        launchParams.mFrame.mNormalBuffer = (math::vec4 *)fbNormal.RawDevicePtr();
        launchParams.mFrame.mAlbedoBuffer = (math::vec4 *)fbAlbedo.RawDevicePtr();

        // and re-set the camera, since aspect may have changed
        setCamera( lastSetCamera );

        // ------------------------------------------------------------------
        OPTIX_CHECK( optixDenoiserSetup( denoiser, 0, aOutputWidth, aOutputHeight, denoiserState.RawDevicePtr(),
                                         denoiserState.SizeAs<uint8_t>(), denoiserScratch.RawDevicePtr(),
                                         denoiserScratch.SizeAs<uint8_t>() ) );
    }

    /*! download the rendered color buffer */
    void RayTracingRenderer::downloadPixels( uint32_t h_pixels[] )
    {
        auto lColors = finalColorBuffer.Fetch<uint32_t>();
        for( uint32_t i = 0; i < lColors.size(); i++ ) h_pixels[i] = lColors[i];
    }

} // namespace SE::Core
