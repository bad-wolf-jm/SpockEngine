#include "Core/Cuda/CudaAssert.h"

#include "LaunchParams.h"
#include "RayTracingRenderer.h"

// #include <optix_function_table_definition.h>

namespace SE::Core
{
    using namespace SE::Cuda;

    extern "C" char embedded_ptx_code[];

#define RECORD_ALIGN __align__( OPTIX_SBT_RECORD_ALIGNMENT )
#define RECORD_HEADER( header ) RECORD_ALIGN char header[OPTIX_SBT_RECORD_HEADER_SIZE]

    struct RECORD_ALIGN sRaygenRecord
    {
        RECORD_HEADER( mDummyHeader );

        void *data;
    };

    struct RECORD_ALIGN sMissRecord
    {
        RECORD_HEADER( mDummyHeader );

        void *data;
    };

    struct RECORD_ALIGN sHitgroupRecord
    {
        RECORD_HEADER( mDummyHeader );

        sTriangleMeshSBTData data;
    };

    RayTracingRenderer::RayTracingRenderer( GraphicContext const &aGraphicContext )
        : mGraphicContext{ aGraphicContext }
    {
        SE::Graphics::OptixDeviceContextObject::Initialize();

        mOptixContext      = New<OptixDeviceContextObject>();
        const int deviceID = 0;
        CUDA_CHECK( SetDevice( deviceID ) );
        CUDA_CHECK( StreamCreate( &stream ) );

        mOptixModule = New<OptixModuleObject>( "optixLaunchParams", embedded_ptx_code, mOptixContext );
        mOptixModule->CreateRayGenGroup( "__raygen__renderFrame" );
        mOptixModule->CreateMissGroup( "__miss__radiance" );
        mOptixModule->CreateHitGroup( "__closesthit__radiance", "__anyhit__radiance" );
        mOptixModule->CreateMissGroup( "__miss__shadow" );
        mOptixModule->CreateHitGroup( "__closesthit__shadow", "__anyhit__shadow" );
        mOptixPipeline = mOptixModule->CreatePipeline();

        mRayTracingParameterBuffer = GPUMemory( sizeof( mRayTracingParameters ) ); //.alloc( sizeof( launchParams ) );
        std::cout << "#osc: context, mOptixModule->mOptixObject, pipeline, etc, all set up ..." << std::endl;
    }

    void RayTracingRenderer::BuildShaderBindingTable()
    {
        mShaderBindingTable = New<OptixShaderBindingTableObject>();

        std::vector<sRaygenRecord> lRaygenRecords =
            mShaderBindingTable->NewRecordType<sRaygenRecord>( mOptixModule->mRayGenProgramGroups );
        mRaygenRecordsBuffer = GPUMemory::Create( lRaygenRecords );
        mShaderBindingTable->BindRayGenRecordTable( mRaygenRecordsBuffer.RawDevicePtr() );

        std::vector<sMissRecord> lMissRecords = mShaderBindingTable->NewRecordType<sMissRecord>( mOptixModule->mMissProgramGroups );
        mMissRecordsBuffer                    = GPUMemory::Create( lMissRecords );
        mShaderBindingTable->BindMissRecordTable<sMissRecord>( mMissRecordsBuffer.RawDevicePtr(),
                                                               mMissRecordsBuffer.SizeAs<sMissRecord>() );

        std::vector<sHitgroupRecord> lHitgroupRecords;
        mScene->ForEach<SE::Core::EntityComponentSystem::Components::sRayTracingTargetComponent,
                        SE::Core::EntityComponentSystem::Components::sStaticMeshComponent>(
            [&]( auto lEntity, auto &lComponent, auto &aMeshComponent )
            {
                for( int lRayTypeID = 0; lRayTypeID < RAY_TYPE_COUNT; lRayTypeID++ )
                {
                    sHitgroupRecord rec =
                        mShaderBindingTable->NewRecordType<sHitgroupRecord>( mOptixModule->mHitProgramGroups[lRayTypeID] );

                    rec.data.mVertexOffset = aMeshComponent.mVertexOffset;
                    rec.data.mIndexOffset  = aMeshComponent.mIndexOffset / 3;
                    if( lEntity.Has<sMaterialComponent>() ) rec.data.mMaterialID = lEntity.Get<sMaterialComponent>().mMaterialID;

                    lHitgroupRecords.push_back( rec );
                }
            } );
        mHitgroupRecordsBuffer = GPUMemory::Create( lHitgroupRecords );
        mShaderBindingTable->BindHitRecordTable<sHitgroupRecord>( mHitgroupRecordsBuffer.RawDevicePtr(),
                                                                  mHitgroupRecordsBuffer.Size() );
    }

    void RayTracingRenderer::Update( Ref<Scene> aWorld )
    {
        mScene                           = aWorld;
        mRayTracingParameters.mSceneRoot = aWorld->GetRayTracingRoot();

        BuildShaderBindingTable();
    }

    /*! render one frame */
    void RayTracingRenderer::Render()
    {
        if( mRayTracingParameters.mFrame.mSize.x == 0 ) return;

        if( !accumulate ) mRayTracingParameters.mFrame.mFrameID = 0;
        mRayTracingParameters.mFrame.mFrameID++;

        mRayTracingParameters.mNumLightSamples = 1;
        mRayTracingParameters.mNumPixelSamples = 1;

        SE::Cuda::GPUExternalMemory lTransformedVertexBuffer( *mScene->mTransformedVertexBuffer,
                                                              mScene->mTransformedVertexBuffer->SizeAs<uint8_t>() );
        SE::Cuda::GPUExternalMemory lIndexBuffer( *mScene->mIndexBuffer, mScene->mIndexBuffer->SizeAs<uint8_t>() );
        mRayTracingParameters.mIndexBuffer  = lIndexBuffer.DataAs<math::uvec3>();
        mRayTracingParameters.mVertexBuffer = lTransformedVertexBuffer.DataAs<VertexData>();

        mRayTracingParameters.mTextures  = mScene->GetMaterialSystem()->GetCudaTextures().DataAs<Cuda::TextureSampler2D::DeviceData>();
        mRayTracingParameters.mMaterials = mScene->GetMaterialSystem()->GetCudaMaterials().DataAs<sShaderMaterial>();

        mRayTracingParameterBuffer.Upload( mRayTracingParameters );

        mOptixPipeline->Launch( stream, mRayTracingParameterBuffer.RawDevicePtr(), mRayTracingParameterBuffer.SizeAs<uint8_t>(),
                                mShaderBindingTable,
                                math::uvec3{ mRayTracingParameters.mFrame.mSize.x, mRayTracingParameters.mFrame.mSize.y, 1 } );

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
            denoiserParams.blendFactor = 1.f / ( mRayTracingParameters.mFrame.mFrameID );
        else
            denoiserParams.blendFactor = 0.0f;

        OptixImage2D inputLayer[3];
        inputLayer[0].data               = fbColor.RawDevicePtr();
        inputLayer[0].width              = mRayTracingParameters.mFrame.mSize.x;
        inputLayer[0].height             = mRayTracingParameters.mFrame.mSize.y;
        inputLayer[0].rowStrideInBytes   = mRayTracingParameters.mFrame.mSize.x * sizeof( float4 );
        inputLayer[0].pixelStrideInBytes = sizeof( float4 );
        inputLayer[0].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        inputLayer[2].data               = fbNormal.RawDevicePtr();
        inputLayer[2].width              = mRayTracingParameters.mFrame.mSize.x;
        inputLayer[2].height             = mRayTracingParameters.mFrame.mSize.y;
        inputLayer[2].rowStrideInBytes   = mRayTracingParameters.mFrame.mSize.x * sizeof( float4 );
        inputLayer[2].pixelStrideInBytes = sizeof( float4 );
        inputLayer[2].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        inputLayer[1].data               = fbAlbedo.RawDevicePtr();
        inputLayer[1].width              = mRayTracingParameters.mFrame.mSize.x;
        inputLayer[1].height             = mRayTracingParameters.mFrame.mSize.y;
        inputLayer[1].rowStrideInBytes   = mRayTracingParameters.mFrame.mSize.x * sizeof( float4 );
        inputLayer[1].pixelStrideInBytes = sizeof( float4 );
        inputLayer[1].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        OptixImage2D outputLayer;
        outputLayer.data               = mDenoisedBuffer.RawDevicePtr();
        outputLayer.width              = mRayTracingParameters.mFrame.mSize.x;
        outputLayer.height             = mRayTracingParameters.mFrame.mSize.y;
        outputLayer.rowStrideInBytes   = mRayTracingParameters.mFrame.mSize.x * sizeof( float4 );
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

        computeFinalPixelColors( mRayTracingParameters, mDenoisedBuffer, mCudaOutputBuffer );

        CUDA_SYNC_CHECK();

        mOutputTexture->TransitionImageLayout( VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        mOutputTexture->CopyBufferToImage( *mOutputBuffer );
        mOutputTexture->TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );
    }

    /*! set camera to render with */
    void RayTracingRenderer::SetView( math::mat4 aViewMatrix )
    {
        ASceneRenderer::SetView( aViewMatrix );
        mRayTracingParameters.mFrame.mFrameID    = 0;
        mRayTracingParameters.mCamera.mPosition  = -math::vec3( mViewMatrix[3] );
        mRayTracingParameters.mCamera.mDirection = -normalize( math::vec3( mViewMatrix[0][2], mViewMatrix[1][2], mViewMatrix[2][2] ) );
        const float cosFovy                      = 0.66f;
        const float aspect = float( mRayTracingParameters.mFrame.mSize.x ) / float( mRayTracingParameters.mFrame.mSize.y );
        mRayTracingParameters.mCamera.mHorizontal =
            cosFovy * aspect *
            math::normalize( math::cross( mRayTracingParameters.mCamera.mDirection,
                                          normalize( math::vec3( mViewMatrix[0][1], mViewMatrix[1][1], mViewMatrix[2][1] ) ) ) );
        mRayTracingParameters.mCamera.mVertical =
            cosFovy * normalize( cross( mRayTracingParameters.mCamera.mHorizontal, mRayTracingParameters.mCamera.mDirection ) );
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

        mDenoisedBuffer.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        fbColor.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        fbNormal.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        fbAlbedo.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        mFinalColorBuffer.Resize( aOutputWidth * aOutputHeight * sizeof( uint32_t ) );

        TextureDescription lTextureCreateInfo{};
        lTextureCreateInfo.Format    = eColorFormat::RGBA8_UNORM;
        lTextureCreateInfo.MipLevels = { Mip{ aOutputWidth, aOutputHeight, 0 } };
        lTextureCreateInfo.Usage     = { TextureUsageFlags::TRANSFER_DESTINATION, TextureUsageFlags::SAMPLED };
        lTextureCreateInfo.Sampled   = true;
        mOutputBuffer                = New<Graphics::Buffer>( mGraphicContext, eBufferBindType::UNKNOWN, false, true, true, false,
                                               aOutputWidth * aOutputHeight * sizeof( uint32_t ) );
        mOutputTexture               = New<Graphics::Texture2D>( mGraphicContext, lTextureCreateInfo );
        mOutputTexture->TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        mCudaOutputBuffer = GPUExternalMemory( *mOutputBuffer, aOutputWidth * aOutputHeight * sizeof( uint32_t ) );
        // update the launch parameters that we'll pass to the optix
        // launch:
        mRayTracingParameters.mFrame.mSize         = math::ivec2{ aOutputWidth, aOutputHeight };
        mRayTracingParameters.mFrame.mColorBuffer  = (math::vec4 *)fbColor.RawDevicePtr();
        mRayTracingParameters.mFrame.mNormalBuffer = (math::vec4 *)fbNormal.RawDevicePtr();
        mRayTracingParameters.mFrame.mAlbedoBuffer = (math::vec4 *)fbAlbedo.RawDevicePtr();

        // and re-set the camera, since aspect may have changed
        SetView( mViewMatrix );

        // ------------------------------------------------------------------
        OPTIX_CHECK( optixDenoiserSetup( denoiser, 0, aOutputWidth, aOutputHeight, denoiserState.RawDevicePtr(),
                                         denoiserState.SizeAs<uint8_t>(), denoiserScratch.RawDevicePtr(),
                                         denoiserScratch.SizeAs<uint8_t>() ) );
    }
} // namespace SE::Core
