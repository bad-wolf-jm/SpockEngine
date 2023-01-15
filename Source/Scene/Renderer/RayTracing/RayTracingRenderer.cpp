#include "Core/CUDA/CudaAssert.h"

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

    RayTracingRenderer::RayTracingRenderer( Ref<VkGraphicContext> aGraphicContext )
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

        mRaygenRecordsBuffer.Dispose();
        mMissRecordsBuffer.Dispose();
        mHitgroupRecordsBuffer.Dispose();

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

                    rec.data.mVertexBuffer = aMeshComponent.mVertexBuffer->DataAs<VertexData>();
                    rec.data.mIndexBuffer  = aMeshComponent.mIndexBuffer->DataAs<math::uvec3>();
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

        mRayTracingParameters.mTextures  = mScene->GetMaterialSystem()->GetCudaTextures().DataAs<Cuda::TextureSampler2D::DeviceData>();
        mRayTracingParameters.mMaterials = mScene->GetMaterialSystem()->GetCudaMaterials().DataAs<sShaderMaterial>();

        mRayTracingParameterBuffer.Upload( mRayTracingParameters );

        mOptixPipeline->Launch( stream, mRayTracingParameterBuffer.RawDevicePtr(), mRayTracingParameterBuffer.SizeAs<uint8_t>(),
                                mShaderBindingTable,
                                math::uvec3{ mRayTracingParameters.mFrame.mSize.x, mRayTracingParameters.mFrame.mSize.y, 1 } );

        denoiserIntensity.Resize( sizeof( float ) );

        OptixDenoiserParams denoiserParams;
        denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;

        denoiserParams.hdrIntensity = denoiserIntensity.RawDevicePtr();
        if( accumulate )
            denoiserParams.blendFactor = 1.f / ( mRayTracingParameters.mFrame.mFrameID );
        else
            denoiserParams.blendFactor = 0.0f;

        OptixImage2D lInputLayer[3];
        lInputLayer[0].data               = fbColor.RawDevicePtr();
        lInputLayer[0].width              = mRayTracingParameters.mFrame.mSize.x;
        lInputLayer[0].height             = mRayTracingParameters.mFrame.mSize.y;
        lInputLayer[0].rowStrideInBytes   = mRayTracingParameters.mFrame.mSize.x * sizeof( float4 );
        lInputLayer[0].pixelStrideInBytes = sizeof( float4 );
        lInputLayer[0].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        lInputLayer[2].data               = fbNormal.RawDevicePtr();
        lInputLayer[2].width              = mRayTracingParameters.mFrame.mSize.x;
        lInputLayer[2].height             = mRayTracingParameters.mFrame.mSize.y;
        lInputLayer[2].rowStrideInBytes   = mRayTracingParameters.mFrame.mSize.x * sizeof( float4 );
        lInputLayer[2].pixelStrideInBytes = sizeof( float4 );
        lInputLayer[2].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        lInputLayer[1].data               = fbAlbedo.RawDevicePtr();
        lInputLayer[1].width              = mRayTracingParameters.mFrame.mSize.x;
        lInputLayer[1].height             = mRayTracingParameters.mFrame.mSize.y;
        lInputLayer[1].rowStrideInBytes   = mRayTracingParameters.mFrame.mSize.x * sizeof( float4 );
        lInputLayer[1].pixelStrideInBytes = sizeof( float4 );
        lInputLayer[1].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        OptixImage2D lOutputLayer;
        lOutputLayer.data               = mDenoisedBuffer.RawDevicePtr();
        lOutputLayer.width              = mRayTracingParameters.mFrame.mSize.x;
        lOutputLayer.height             = mRayTracingParameters.mFrame.mSize.y;
        lOutputLayer.rowStrideInBytes   = mRayTracingParameters.mFrame.mSize.x * sizeof( float4 );
        lOutputLayer.pixelStrideInBytes = sizeof( float4 );
        lOutputLayer.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        // -------------------------------------------------------
        if( denoiserOn )
        {
            OPTIX_CHECK( optixDenoiserComputeIntensity( denoiser, 0, &lInputLayer[0], (CUdeviceptr)denoiserIntensity.RawDevicePtr(),
                                                        (CUdeviceptr)denoiserScratch.RawDevicePtr(),
                                                        denoiserScratch.SizeAs<uint8_t>() ) );

            OptixDenoiserGuideLayer denoiserGuideLayer{};
            denoiserGuideLayer.albedo = lInputLayer[1];
            denoiserGuideLayer.normal = lInputLayer[2];

            OptixDenoiserLayer denoiserLayer{};
            denoiserLayer.input  = lInputLayer[0];
            denoiserLayer.output = lOutputLayer;

            OPTIX_CHECK( optixDenoiserInvoke( denoiser, 0, &denoiserParams, denoiserState.RawDevicePtr(),
                                              denoiserState.SizeAs<uint8_t>(), &denoiserGuideLayer, &denoiserLayer, 1, 0, 0,
                                              denoiserScratch.RawDevicePtr(), denoiserScratch.SizeAs<uint8_t>() ) );
        }
        else
        {
            cudaMemcpy( (void *)lOutputLayer.data, (void *)lInputLayer[0].data,
                        lOutputLayer.width * lOutputLayer.height * sizeof( float4 ), cudaMemcpyDeviceToDevice );
        }
        computeFinalPixelColors( mRayTracingParameters, mDenoisedBuffer, *mOutputBuffer );

        CUDA_SYNC_CHECK();

        mOutputTexture->TransitionImageLayout( VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
        mOutputTexture->SetPixelData( mOutputBuffer );
        mOutputTexture->TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );
    }

    /*! set camera to render with */
    void RayTracingRenderer::SetView( math::mat4 aViewMatrix )
    {
        ASceneRenderer::SetView( math::Inverse( aViewMatrix ) );

        mRayTracingParameters.mFrame.mFrameID    = 0;
        mRayTracingParameters.mCamera.mPosition  = math::vec3( mViewMatrix[3] );
        mRayTracingParameters.mCamera.mDirection = normalize( math::vec3( mViewMatrix[0][2], mViewMatrix[1][2], mViewMatrix[2][2] ) );
        const float cosFovy                      = 0.66f;
        const float aspect = float( mRayTracingParameters.mFrame.mSize.x ) / float( mRayTracingParameters.mFrame.mSize.y );
        mRayTracingParameters.mCamera.mHorizontal =
            cosFovy * aspect *
            math::normalize( math::cross( mRayTracingParameters.mCamera.mDirection, math::vec3( 0.0f, 1.0f, 0.0f ) ) );
        mRayTracingParameters.mCamera.mVertical = cosFovy * math::normalize( math::cross( mRayTracingParameters.mCamera.mHorizontal,
                                                                                          mRayTracingParameters.mCamera.mDirection ) );
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
        OPTIX_CHECK( optixDenoiserCreate( mOptixContext->mOptixObject, OPTIX_DENOISER_MODEL_KIND_LDR, &denoiserOptions, &denoiser ) );

        // .. then compute and allocate memory resources for the denoiser
        OptixDenoiserSizes denoiserReturnSizes;
        OPTIX_CHECK( optixDenoiserComputeMemoryResources( denoiser, aOutputWidth, aOutputHeight, &denoiserReturnSizes ) );

        denoiserScratch.Resize(
            std::max( denoiserReturnSizes.withOverlapScratchSizeInBytes, denoiserReturnSizes.withoutOverlapScratchSizeInBytes ) );
        denoiserState.Resize( denoiserReturnSizes.stateSizeInBytes );

        // ------------------------------------------------------------------
        // resize our cuda frame buffer

        mDenoisedBuffer.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        fbColor.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        fbNormal.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        fbAlbedo.Resize( aOutputWidth * aOutputHeight * sizeof( float4 ) );
        mFinalColorBuffer.Resize( aOutputWidth * aOutputHeight * sizeof( uint32_t ) );

        sTextureCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mFormat    = eColorFormat::RGBA8_UNORM;
        lTextureCreateInfo.mMipLevels = 1;
        lTextureCreateInfo.mWidth     = aOutputWidth;
        lTextureCreateInfo.mHeight    = aOutputHeight;
        lTextureCreateInfo.mDepth     = 1;
        mOutputTexture                = New<Graphics::VkTexture2D>( mGraphicContext, lTextureCreateInfo, 1, false, false, true, true );
        mOutputTexture->TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        mOutputBuffer = New<Graphics::VkGpuBuffer>( mGraphicContext, eBufferType::UNKNOWN, false, false, true, false,
                                                    aOutputWidth * aOutputHeight * sizeof( uint32_t ) );

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
