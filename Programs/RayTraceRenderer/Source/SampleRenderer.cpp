#include "Core/CUDA/CudaAssert.h"

#include "LaunchParams.h"
#include "SampleRenderer.h"

#include <optix_function_table_definition.h>

namespace osc
{
    using namespace SE::Cuda;
    using namespace SE::Core;

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
    SampleRenderer::SampleRenderer( const Model *model, const QuadLight &light )
        : model( model )
    {
        SE::Graphics::OptixDeviceContextObject::Initialize();

        std::cout << GDT_TERMINAL_GREEN << "#osc: successfully initialized optix... yay!" << GDT_TERMINAL_DEFAULT << std::endl;

        launchParams.mLight.mOrigin = light.mOrigin;
        launchParams.mLight.mDu     = light.mDu;
        launchParams.mLight.mDv     = light.mDv;
        launchParams.mLight.mPower  = light.mPower;

        std::cout << "#osc: creating optix context ..." << std::endl;
        // createContext();
        mOptixContext      = New<OptixDeviceContextObject>();
        const int deviceID = 0;
        CUDA_CHECK( SetDevice( deviceID ) );
        CUDA_CHECK( StreamCreate( &stream ) );

        std::cout << "#osc: setting up module and pipeline ..." << std::endl;
        // createModule();
        mOptixModule = New<OptixModuleObject>( "optixLaunchParams", embedded_ptx_code, mOptixContext );
        mOptixModule->CreateRayGenGroup( "__raygen__renderFrame" );
        mOptixModule->CreateMissGroup( "__miss__radiance" );
        mOptixModule->CreateHitGroup( "__closesthit__radiance", "__anyhit__radiance" );
        mOptixModule->CreateMissGroup( "__miss__shadow" );
        mOptixModule->CreateHitGroup( "__closesthit__shadow", "__anyhit__shadow" );
        mOptixPipeline = mOptixModule->CreatePipeline();

        createTextures();

        launchParams.mSceneRoot = buildAccel();

        std::cout << "#osc: building SBT ..." << std::endl;
        buildSBT();

        launchParamsBuffer = GPUMemory( sizeof( launchParams ) ); //.alloc( sizeof( launchParams ) );
        std::cout << "#osc: context, mOptixModule->mOptixObject, pipeline, etc, all set up ..." << std::endl;

        std::cout << GDT_TERMINAL_GREEN;
        std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
        std::cout << GDT_TERMINAL_DEFAULT;
    }

    void SampleRenderer::createTextures()
    {
        int numTextures = (int)model->mTextures.size();

        textureArrays.resize( numTextures );
        textureObjects.resize( numTextures );

        for( int textureID = 0; textureID < numTextures; textureID++ )
        {
            auto mTexture = model->mTextures[textureID];

            cudaResourceDesc res_desc = {};

            cudaChannelFormatDesc channel_desc;
            int32_t               width         = mTexture->mResolution.x;
            int32_t               height        = mTexture->mResolution.y;
            int32_t               numComponents = 4;
            int32_t               pitch         = width * numComponents * sizeof( uint8_t );
            channel_desc                        = cudaCreateChannelDesc<uchar4>();

            cudaArray_t &pixelArray = textureArrays[textureID];
            CUDA_CHECK( MallocArray( &pixelArray, &channel_desc, width, height ) );

            CUDA_CHECK( Memcpy2DToArray( pixelArray,
                                         /* offset */ 0, 0, mTexture->mPixel, pitch, pitch, height, cudaMemcpyHostToDevice ) );

            res_desc.resType         = cudaResourceTypeArray;
            res_desc.res.array.array = pixelArray;

            cudaTextureDesc tex_desc     = {};
            tex_desc.addressMode[0]      = cudaAddressModeWrap;
            tex_desc.addressMode[1]      = cudaAddressModeWrap;
            tex_desc.filterMode          = cudaFilterModeLinear;
            tex_desc.readMode            = cudaReadModeNormalizedFloat;
            tex_desc.normalizedCoords    = 1;
            tex_desc.maxAnisotropy       = 1;
            tex_desc.maxMipmapLevelClamp = 99;
            tex_desc.minMipmapLevelClamp = 0;
            tex_desc.mipmapFilterMode    = cudaFilterModePoint;
            tex_desc.borderColor[0]      = 1.0f;
            tex_desc.sRGB                = 0;

            // Create mTexture object
            cudaTextureObject_t cuda_tex = 0;
            CUDA_CHECK( CreateTextureObject( &cuda_tex, &res_desc, &tex_desc, nullptr ) );
            textureObjects[textureID] = cuda_tex;
        }
    }

    OptixTraversableHandle SampleRenderer::buildAccel()
    {
        mScene = New<OptixScene>( mOptixContext );

        const int numMeshes = (int)model->mMeshes.size();
        mVertices.resize( numMeshes );
        mIndices.resize( numMeshes );
        mTexCoords.resize( numMeshes );
        mNormals.resize( numMeshes );

        for( int meshID = 0; meshID < numMeshes; meshID++ )
        {
            // upload the model to the device: the builder
            TriangleMesh &mesh = *model->mMeshes[meshID];
            mVertices[meshID]  = GPUMemory::Create<math::vec3>( mesh.mVertex );
            mIndices[meshID]   = GPUMemory::Create<math::ivec3>( mesh.mIndex );

            if( !mesh.mNormal.empty() ) mNormals[meshID] = GPUMemory::Create<math::vec3>( mesh.mNormal );
            if( !mesh.mTexCoord.empty() ) mTexCoords[meshID] = GPUMemory::Create<math::vec2>( mesh.mTexCoord );

            mScene->AddGeometry<math::vec3>( mVertices[meshID], mIndices[meshID], 0, (int)mesh.mVertex.size(), 0,
                                             (int)mesh.mIndex.size() * 3 );
        }

        mScene->Build();

        return mScene->mOptixObject;
    }

    static void context_log_cb( unsigned int level, const char *tag, const char *message, void * )
    {
        fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
    }

    /*! constructs the shader binding table */
    void SampleRenderer::buildSBT()
    {
        mShaderBindingTable = New<OptixShaderBindingTableObject>();

        std::vector<RaygenRecord> raygenRecords =
            mShaderBindingTable->NewRecordType<RaygenRecord>( mOptixModule->mRayGenProgramGroups );
        raygenRecordsBuffer = GPUMemory::Create( raygenRecords );
        mShaderBindingTable->BindRayGenRecordTable( raygenRecordsBuffer.RawDevicePtr() );

        std::vector<MissRecord> missRecords = mShaderBindingTable->NewRecordType<MissRecord>( mOptixModule->mMissProgramGroups );
        missRecordsBuffer                   = GPUMemory::Create( missRecords );
        mShaderBindingTable->BindMissRecordTable<MissRecord>( missRecordsBuffer.RawDevicePtr(),
                                                              missRecordsBuffer.SizeAs<MissRecord>() );

        int numObjects = (int)model->mMeshes.size();

        std::vector<HitgroupRecord> hitgroupRecords;
        for( int meshID = 0; meshID < numObjects; meshID++ )
        {
            for( int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++ )
            {
                auto mesh = model->mMeshes[meshID];

                HitgroupRecord rec = mShaderBindingTable->NewRecordType<HitgroupRecord>( mOptixModule->mHitProgramGroups[rayID] );
                rec.data.mColor    = mesh->mDiffuse;
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
        }
        hitgroupRecordsBuffer = GPUMemory::Create( hitgroupRecords );
        mShaderBindingTable->BindHitRecordTable<HitgroupRecord>( hitgroupRecordsBuffer.RawDevicePtr(), hitgroupRecordsBuffer.Size() );
    }

    /*! render one frame */
    void SampleRenderer::render()
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
        denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
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

        // -------------------------------------------------------
        OptixImage2D inputLayer[3];
        inputLayer[0].data = fbColor.RawDevicePtr();
        /// Width of the image (in pixels)
        inputLayer[0].width = launchParams.mFrame.mSize.x;
        /// Height of the image (in pixels)
        inputLayer[0].height = launchParams.mFrame.mSize.y;
        /// Stride between subsequent rows of the image (in bytes).
        inputLayer[0].rowStrideInBytes = launchParams.mFrame.mSize.x * sizeof( float4 );
        /// Stride between subsequent pixels of the image (in bytes).
        /// For now, only 0 or the value that corresponds to a dense packing of pixels
        /// (no gaps) is supported.
        inputLayer[0].pixelStrideInBytes = sizeof( float4 );
        /// Pixel format.
        inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

        // ..................................................................
        inputLayer[2].data = fbNormal.RawDevicePtr();
        /// Width of the image (in pixels)
        inputLayer[2].width = launchParams.mFrame.mSize.x;
        /// Height of the image (in pixels)
        inputLayer[2].height = launchParams.mFrame.mSize.y;
        /// Stride between subsequent rows of the image (in bytes).
        inputLayer[2].rowStrideInBytes = launchParams.mFrame.mSize.x * sizeof( float4 );
        /// Stride between subsequent pixels of the image (in bytes).
        /// For now, only 0 or the value that corresponds to a dense packing of pixels
        /// (no gaps) is supported.
        inputLayer[2].pixelStrideInBytes = sizeof( float4 );
        /// Pixel format.
        inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;

        // ..................................................................
        inputLayer[1].data = fbAlbedo.RawDevicePtr();
        /// Width of the image (in pixels)
        inputLayer[1].width = launchParams.mFrame.mSize.x;
        /// Height of the image (in pixels)
        inputLayer[1].height = launchParams.mFrame.mSize.y;
        /// Stride between subsequent rows of the image (in bytes).
        inputLayer[1].rowStrideInBytes = launchParams.mFrame.mSize.x * sizeof( float4 );
        /// Stride between subsequent pixels of the image (in bytes).
        /// For now, only 0 or the value that corresponds to a dense packing of pixels
        /// (no gaps) is supported.
        inputLayer[1].pixelStrideInBytes = sizeof( float4 );
        /// Pixel format.
        inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;

        // -------------------------------------------------------
        OptixImage2D outputLayer;
        outputLayer.data = denoisedBuffer.RawDevicePtr();
        /// Width of the image (in pixels)
        outputLayer.width = launchParams.mFrame.mSize.x;
        /// Height of the image (in pixels)
        outputLayer.height = launchParams.mFrame.mSize.y;
        /// Stride between subsequent rows of the image (in bytes).
        outputLayer.rowStrideInBytes = launchParams.mFrame.mSize.x * sizeof( float4 );
        /// Stride between subsequent pixels of the image (in bytes).
        /// For now, only 0 or the value that corresponds to a dense packing of pixels
        /// (no gaps) is supported.
        outputLayer.pixelStrideInBytes = sizeof( float4 );
        /// Pixel format.
        outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        // -------------------------------------------------------
        if( denoiserOn )
        {
            OPTIX_CHECK( optixDenoiserComputeIntensity( denoiser,
                                                        /*stream*/ 0, &inputLayer[0], (CUdeviceptr)denoiserIntensity.RawDevicePtr(),
                                                        (CUdeviceptr)denoiserScratch.RawDevicePtr(),
                                                        denoiserScratch.SizeAs<uint8_t>() ) );

#if OPTIX_VERSION >= 70300
            OptixDenoiserGuideLayer denoiserGuideLayer = {};
            denoiserGuideLayer.albedo                  = inputLayer[1];
            denoiserGuideLayer.normal                  = inputLayer[2];

            OptixDenoiserLayer denoiserLayer = {};
            denoiserLayer.input              = inputLayer[0];
            denoiserLayer.output             = outputLayer;

            OPTIX_CHECK( optixDenoiserInvoke( denoiser,
                                              /*stream*/ 0, &denoiserParams, denoiserState.RawDevicePtr(),
                                              denoiserState.SizeAs<uint8_t>(), &denoiserGuideLayer, &denoiserLayer, 1,
                                              /*inputOffsetX*/ 0,
                                              /*inputOffsetY*/ 0, denoiserScratch.RawDevicePtr(),
                                              denoiserScratch.SizeAs<uint8_t>() ) );
#else
            OPTIX_CHECK( optixDenoiserInvoke(
                denoiser,
                /*stream*/ 0, &denoiserParams, denoiserState.RawDevicePtr(), denoiserState.SizeAs<uint8_t>(), &inputLayer[0], 2,
                /*inputOffsetX*/ 0,
                /*inputOffsetY*/ 0, &outputLayer, denoiserScratch.RawDevicePtr(), denoiserScratch.SizeAs<uint8_t>() ) );
#endif
        }
        else
        {
            cudaMemcpy( (void *)outputLayer.data, (void *)inputLayer[0].data,
                        outputLayer.width * outputLayer.height * sizeof( float4 ), cudaMemcpyDeviceToDevice );
        }
        computeFinalPixelColors();

        // sync - make sure the frame is rendered before we download and
        // display (obviously, for a high-performance application you
        // want to use streams and double-buffering, but for this simple
        // example, this will have to do)
        CUDA_SYNC_CHECK();
    }

    /*! set camera to render with */
    void SampleRenderer::setCamera( const Camera &camera )
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
    void SampleRenderer::resize( const math::ivec2 &newSize )
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
        OPTIX_CHECK( optixDenoiserComputeMemoryResources( denoiser, newSize.x, newSize.y, &denoiserReturnSizes ) );

#if OPTIX_VERSION < 70100
        denoiserScratch.Resize( denoiserReturnSizes.recommendedScratchSizeInBytes );
#else
        denoiserScratch.Resize(
            std::max( denoiserReturnSizes.withOverlapScratchSizeInBytes, denoiserReturnSizes.withoutOverlapScratchSizeInBytes ) );
#endif
        denoiserState.Resize( denoiserReturnSizes.stateSizeInBytes );

        // ------------------------------------------------------------------
        // resize our cuda frame buffer

        denoisedBuffer.Resize( newSize.x * newSize.y * sizeof( float4 ) );
        fbColor.Resize( newSize.x * newSize.y * sizeof( float4 ) );
        fbNormal.Resize( newSize.x * newSize.y * sizeof( float4 ) );
        fbAlbedo.Resize( newSize.x * newSize.y * sizeof( float4 ) );
        finalColorBuffer.Resize( newSize.x * newSize.y * sizeof( uint32_t ) );

        // update the launch parameters that we'll pass to the optix
        // launch:
        launchParams.mFrame.mSize         = newSize;
        launchParams.mFrame.mColorBuffer  = (math::vec4 *)fbColor.RawDevicePtr();
        launchParams.mFrame.mNormalBuffer = (math::vec4 *)fbNormal.RawDevicePtr();
        launchParams.mFrame.mAlbedoBuffer = (math::vec4 *)fbAlbedo.RawDevicePtr();

        // and re-set the camera, since aspect may have changed
        setCamera( lastSetCamera );

        // ------------------------------------------------------------------
        OPTIX_CHECK( optixDenoiserSetup( denoiser, 0, newSize.x, newSize.y, denoiserState.RawDevicePtr(),
                                         denoiserState.SizeAs<uint8_t>(), denoiserScratch.RawDevicePtr(),
                                         denoiserScratch.SizeAs<uint8_t>() ) );
    }

    /*! download the rendered color buffer */
    void SampleRenderer::downloadPixels( uint32_t h_pixels[] )
    {
        auto lColors = finalColorBuffer.Fetch<uint32_t>();
        for( uint32_t i = 0; i < lColors.size(); i++ ) h_pixels[i] = lColors[i];
    }

} // namespace osc
