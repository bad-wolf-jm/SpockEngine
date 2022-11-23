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

#pragma once

#include "Core/Memory.h"

// our own classes, partly shared between host and device
#include "LaunchParams.h"
#include "Model.h"

#include "Core/Cuda/CudaBuffer.h"
#include "Core/Math/Types.h"

#include "Core/Optix/OptixContext.h"
#include "Core/Optix/OptixModule.h"
#include "Core/Optix/OptixShaderBindingTable.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{
    using namespace SE::Cuda;
    using namespace SE::Core;
    using namespace SE::Graphics;

    struct Camera
    {
        /*! camera position - *from* where we are looking */
        math::vec3 from;
        /*! which point we are looking *at* */
        math::vec3 at;
        /*! general up-vector */
        math::vec3 up;
    };

    /*! a sample OptiX-7 renderer that demonstrates how to set up
        context, module, programs, pipeline, SBT, etc, and perform a
        valid launch that renders some pixel (using a simple test
        pattern, in this case */
    class SampleRenderer
    {
        // ------------------------------------------------------------------
        // publicly accessible interface
        // ------------------------------------------------------------------
      public:
        /*! constructor - performs all setup, including initializing
          optix, creates module, pipeline, programs, SBT, etc. */
        SampleRenderer( const Model *model, const QuadLight &light );

        /*! render one frame */
        void render();

        /*! resize frame buffer to given resolution */
        void resize( const math::ivec2 &newSize );

        /*! download the rendered color buffer */
        void downloadPixels( uint32_t h_pixels[] );

        /*! set camera to render with */
        void setCamera( const Camera &camera );

        bool denoiserOn = true;
        bool accumulate = true;

      protected:
        // ------------------------------------------------------------------
        // internal helper functions
        // ------------------------------------------------------------------

        /*! runs a cuda kernel that performs gamma correction and float4-to-rgba
         * conversion */
        void computeFinalPixelColors();

        /*! helper function that initializes optix and checks for errors */
        void initOptix();

        /*! creates and configures a optix device context (in this simple
          example, only for the primary GPU device) */
        void createContext();

        /*! creates the module that contains all the programs we are going
          to use. in this simple example, we use a single module from a
          single .cu file, using a single embedded ptx string */
        void createModule();

        // /*! does all setup for the raygen program(s) we are going to use */
        // void createRaygenPrograms();

        // /*! does all setup for the miss program(s) we are going to use */
        // void createMissPrograms();

        // /*! does all setup for the hitgroup program(s) we are going to use */
        // void createHitgroupPrograms();

        // /*! assembles the full pipeline of all programs */
        // void createPipeline();

        /*! constructs the shader binding table */
        void buildSBT();

        /*! build an acceleration structure for the given triangle mesh */
        OptixTraversableHandle buildAccel();

        /*! upload textures, and create cuda mTexture objects for them */
        void createTextures();

      protected:
        /*! @{ CUDA device context and stream that optix pipeline will run
            on, as well as device properties for this device */
        CUcontext      cudaContext;
        CUstream       stream;
        cudaDeviceProp deviceProps;
        /*! @} */

        //! the optix context that our pipeline will run in.
        // OptixDeviceContext optixContext;

        /*! @{ the pipeline we're building */
        OptixPipeline               pipeline;
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        OptixPipelineLinkOptions    pipelineLinkOptions    = {};
        /*! @} */

        /*! @{ the module that contains out device programs */
        OptixModule               module;
        OptixModuleCompileOptions moduleCompileOptions = {};
        /* @} */

        /*! vector of all our program(group)s, and the SBT built around
            them */
        std::vector<OptixProgramGroup> raygenPGs;
        GPUMemory                      raygenRecordsBuffer;
        std::vector<OptixProgramGroup> missPGs;
        GPUMemory                      missRecordsBuffer;
        std::vector<OptixProgramGroup> hitgroupPGs;
        GPUMemory                      hitgroupRecordsBuffer;
        OptixShaderBindingTable        sbt = {};

        /*! @{ our launch parameters, on the host, and the buffer to store
            them on the device */
      public:
        sLaunchParams launchParams;

      protected:
        GPUMemory launchParamsBuffer;
        /*! @} */

        /*! the color buffer we use during _rendering_, which is a bit
            larger than the actual displayed frame buffer (to account for
            the border), and in float4 format (the denoiser requires
            floats) */
        GPUMemory fbColor;
        GPUMemory fbNormal;
        GPUMemory fbAlbedo;

        /*! output of the denoiser pass, in float4 */
        GPUMemory denoisedBuffer;

        /* the actual final color buffer used for display, in rgba8 */
        GPUMemory finalColorBuffer;

        OptixDenoiser denoiser = nullptr;
        GPUMemory     denoiserScratch;
        GPUMemory     denoiserState;
        GPUMemory     denoiserIntensity;

        /*! the camera we are to render with. */
        Camera lastSetCamera;

        /*! the model we are going to trace rays against */
        const Model *model;

        std::vector<GPUMemory> mVertices;
        std::vector<GPUMemory> mIndices;
        std::vector<GPUMemory> mNormals;
        std::vector<GPUMemory> mTexCoords;

        //! buffer that keeps the (final, compacted) accel structure
        GPUMemory asBuffer;

        /*! @{ one mTexture object and pixel array per used mTexture */
        std::vector<cudaArray_t>         textureArrays;
        std::vector<cudaTextureObject_t> textureObjects;
        /*! @} */

        Ref<OptixDeviceContextObject> mOptixContext = nullptr;
        Ref<OptixModuleObject> mOptixModule = nullptr;
        Ref<OptixPipelineObject> mOptixPipeline = nullptr;
        Ref<OptixShaderBindingTableObject> mShaderBindingTable = nullptr;

    };

} // namespace osc
