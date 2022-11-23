// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "Core/Cuda/CudaAssert.h"

#include "LaunchParams.h"
#include "SampleRenderer.h"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

/*! \namespace osc - Optix Siggraph Course */
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
        initOptix();

        launchParams.mLight.mOrigin = light.mOrigin;
        launchParams.mLight.mDu     = light.mDu;
        launchParams.mLight.mDv     = light.mDv;
        launchParams.mLight.mPower  = light.mPower;

        std::cout << "#osc: creating optix context ..." << std::endl;
        createContext();

        std::cout << "#osc: setting up mOptixModule->mOptixObject ..." << std::endl;
        createModule();

        // std::cout << "#osc: creating raygen programs ..." << std::endl;
        // createRaygenPrograms();
        // std::cout << "#osc: creating miss programs ..." << std::endl;
        // createMissPrograms();
        // std::cout << "#osc: creating hitgroup programs ..." << std::endl;
        // createHitgroupPrograms();

        std::cout << "#osc: setting up optix pipeline ..." << std::endl;
        // createPipeline();

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
        const int numMeshes = (int)model->mMeshes.size();
        mVertices.resize( numMeshes );
        mIndices.resize( numMeshes );
        mTexCoords.resize( numMeshes );
        mNormals.resize( numMeshes );

        OptixTraversableHandle asHandle{ 0 };

        // ==================================================================
        // triangle inputs
        // ==================================================================
        std::vector<OptixBuildInput> triangleInput( numMeshes );
        std::vector<uint32_t>        triangleInputFlags( numMeshes );

        for( int meshID = 0; meshID < numMeshes; meshID++ )
        {
            // upload the model to the device: the builder
            TriangleMesh &mesh = *model->mMeshes[meshID];
            mVertices[meshID]  = GPUMemory::Create<math::vec3>( mesh.mVertex );
            mIndices[meshID]   = GPUMemory::Create<math::ivec3>( mesh.mIndex );

            if( !mesh.mNormal.empty() ) mNormals[meshID] = GPUMemory::Create<math::vec3>( mesh.mNormal );
            if( !mesh.mTexCoord.empty() ) mTexCoords[meshID] = GPUMemory::Create<math::vec2>( mesh.mTexCoord );

            triangleInput[meshID]      = {};
            triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            triangleInput[meshID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof( math::vec3 );
            triangleInput[meshID].triangleArray.numVertices         = (int)mesh.mVertex.size();
            triangleInput[meshID].triangleArray.vertexBuffers       = mVertices[meshID].RawDevicePtrP(); //&d_vertices[meshID];

            triangleInput[meshID].triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof( math::ivec3 );
            triangleInput[meshID].triangleArray.numIndexTriplets   = (int)mesh.mIndex.size();
            triangleInput[meshID].triangleArray.indexBuffer        = mIndices[meshID].RawDevicePtr(); //  d_indices[meshID];

            triangleInputFlags[meshID] = 0;

            // in this example we have one SBT entry, and no per-primitive
            // materials:
            triangleInput[meshID].triangleArray.flags                       = &triangleInputFlags[meshID];
            triangleInput[meshID].triangleArray.numSbtRecords               = 1;
            triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer        = 0;
            triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0;
            triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
        }
        // ==================================================================
        // BLAS setup
        // ==================================================================

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.motionOptions.numKeys  = 1;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes blasBufferSizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( mOptixContext->mOptixObject, &accelOptions, triangleInput.data(), (int)numMeshes,
                                                   &blasBufferSizes ) );

        // ==================================================================
        // prepare compaction
        // ==================================================================

        GPUMemory compactedSizeBuffer( sizeof( uint64_t ) );
        // compactedSizeBuffer.alloc( sizeof( uint64_t ) );

        OptixAccelEmitDesc emitDesc;
        emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = compactedSizeBuffer.RawDevicePtr();

        // ==================================================================
        // execute build (main stage)
        // ==================================================================

        GPUMemory tempBuffer( blasBufferSizes.tempSizeInBytes );
        // tempBuffer.alloc( blasBufferSizes.tempSizeInBytes );

        GPUMemory outputBuffer( blasBufferSizes.outputSizeInBytes );
        // outputBuffer.alloc( blasBufferSizes.outputSizeInBytes );

        OPTIX_CHECK( optixAccelBuild( mOptixContext->mOptixObject, 0, &accelOptions, triangleInput.data(), (int)numMeshes,
                                      tempBuffer.RawDevicePtr(), tempBuffer.SizeAs<uint8_t>(), outputBuffer.RawDevicePtr(),
                                      outputBuffer.SizeAs<uint8_t>(), &asHandle, &emitDesc, 1 ) );
        CUDA_SYNC_CHECK();

        // ==================================================================
        // perform compaction
        // ==================================================================
        uint64_t compactedSize = compactedSizeBuffer.Fetch<uint64_t>()[0];
        // compactedSizeBuffer.download( &compactedSize, 1 );

        asBuffer = GPUMemory( compactedSize ); //.alloc( compactedSize );
        OPTIX_CHECK( optixAccelCompact( mOptixContext->mOptixObject, 0, asHandle, asBuffer.RawDevicePtr(), asBuffer.SizeAs<uint8_t>(),
                                        &asHandle ) );
        CUDA_SYNC_CHECK();

        // ==================================================================
        // aaaaaand .... clean up
        // ==================================================================
        outputBuffer.Dispose(); // << the UNcompacted, temporary output buffer
        tempBuffer.Dispose();
        compactedSizeBuffer.Dispose();

        return asHandle;
    }

    /*! helper function that initializes optix and checks for errors */
    void SampleRenderer::initOptix()
    {
        SE::Graphics::OptixDeviceContextObject::Initialize();

        std::cout << GDT_TERMINAL_GREEN << "#osc: successfully initialized optix... yay!" << GDT_TERMINAL_DEFAULT << std::endl;
    }

    static void context_log_cb( unsigned int level, const char *tag, const char *message, void * )
    {
        fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
    }

    /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void SampleRenderer::createContext()
    {
        mOptixContext = New<OptixDeviceContextObject>();

        const int deviceID = 0;
        CUDA_CHECK( SetDevice( deviceID ) );
        CUDA_CHECK( StreamCreate( &stream ) );
    }

    /*! creates the mOptixModule->mOptixObject that contains all the programs we are going
      to use. in this simple example, we use a single mOptixModule->mOptixObject from a
      single .cu file, using a single embedded ptx string */
    void SampleRenderer::createModule()
    {
        mOptixModule = New<OptixModuleObject>( "optixLaunchParams", embedded_ptx_code, mOptixContext );
 
        mOptixModule->CreateRayGenGroup( "__raygen__renderFrame" );

        mOptixModule->CreateMissGroup( "__miss__radiance" );
        mOptixModule->CreateHitGroup( "__closesthit__radiance", "__anyhit__radiance" );

        mOptixModule->CreateMissGroup( "__miss__shadow" );
        mOptixModule->CreateHitGroup( "__closesthit__shadow", "__anyhit__shadow" );

        mOptixPipeline = mOptixModule->CreatePipeline();
    }

 
    /*! constructs the shader binding table */
    void SampleRenderer::buildSBT()
    {
        // ------------------------------------------------------------------
        // build raygen records
        // ------------------------------------------------------------------
        std::vector<RaygenRecord> raygenRecords;
        for( int i = 0; i < mOptixModule->mRayGenProgramGroups.size(); i++ )
        {
            RaygenRecord rec;
            OPTIX_CHECK( optixSbtRecordPackHeader( mOptixModule->mRayGenProgramGroups[i]->mOptixObject, &rec ) );
            rec.data = nullptr; /* for now ... */
            raygenRecords.push_back( rec );
        }
        raygenRecordsBuffer = GPUMemory::Create( raygenRecords );
        sbt.raygenRecord    = raygenRecordsBuffer.RawDevicePtr();

        // ------------------------------------------------------------------
        // build miss records
        // ------------------------------------------------------------------
        std::vector<MissRecord> missRecords;
        for( int i = 0; i < mOptixModule->mMissProgramGroups.size(); i++ )
        {
            MissRecord rec;
            OPTIX_CHECK( optixSbtRecordPackHeader( mOptixModule->mMissProgramGroups[i]->mOptixObject, &rec ) );
            rec.data = nullptr; /* for now ... */
            missRecords.push_back( rec );
        }
        missRecordsBuffer           = GPUMemory::Create( missRecords );
        sbt.missRecordBase          = missRecordsBuffer.RawDevicePtr();
        sbt.missRecordStrideInBytes = sizeof( MissRecord );
        sbt.missRecordCount         = (int)missRecords.size();

        // ------------------------------------------------------------------
        // build hitgroup records
        // ------------------------------------------------------------------
        int                         numObjects = (int)model->mMeshes.size();
        std::vector<HitgroupRecord> hitgroupRecords;
        for( int meshID = 0; meshID < numObjects; meshID++ )
        {
            for( int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++ )
            {
                auto mesh = model->mMeshes[meshID];

                HitgroupRecord rec;
                OPTIX_CHECK( optixSbtRecordPackHeader( mOptixModule->mHitProgramGroups[rayID]->mOptixObject, &rec ) );
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
        }
        hitgroupRecordsBuffer           = GPUMemory::Create( hitgroupRecords );
        sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.RawDevicePtr();
        sbt.hitgroupRecordStrideInBytes = sizeof( HitgroupRecord );
        sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
    }

    /*! render one frame */
    void SampleRenderer::render()
    {
        // sanity check: make sure we launch only after first resize is
        // already done:
        if( launchParams.mFrame.mSize.x == 0 ) return;

        if( !accumulate ) launchParams.mFrame.mFrameID = 0;
        launchParamsBuffer.Upload( launchParams );
        launchParams.mFrame.mFrameID++;

        launchParams.mNumLightSamples = 1;
        launchParams.mNumPixelSamples = 1;

        OPTIX_CHECK( optixLaunch( mOptixPipeline->mOptixObject, stream, launchParamsBuffer.RawDevicePtr(),
                                  launchParamsBuffer.SizeAs<uint8_t>(), &sbt, launchParams.mFrame.mSize.x, launchParams.mFrame.mSize.y,
                                  1 ) );

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
        // finalColorBuffer.download( h_pixels, launchParams.mFrame.mSize.x * launchParams.mFrame.mSize.y );
    }

} // namespace osc
