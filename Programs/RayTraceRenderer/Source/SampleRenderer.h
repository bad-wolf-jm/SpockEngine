#pragma once

#include "Core/Memory.h"

// our own classes, partly shared between host and device
#include "LaunchParams.h"
#include "Model.h"

#include "Core/CUDA/Array/CudaBuffer.h"
#include "Core/Math/Types.h"

#include "Core/Optix/OptixAccelerationStructure.h"
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

        /*! constructs the shader binding table */
        void buildSBT();

        /*! build an acceleration structure for the given triangle mesh */
        OptixTraversableHandle buildAccel();

        /*! upload textures, and create cuda mTexture objects for them */
        void createTextures();

      protected:
        CUcontext      cudaContext;
        CUstream       stream;
        cudaDeviceProp deviceProps;
 
        GPUMemory                      raygenRecordsBuffer;
        GPUMemory                      missRecordsBuffer;
        GPUMemory                      hitgroupRecordsBuffer;
 
      public:
        sLaunchParams launchParams;

      protected:
        GPUMemory launchParamsBuffer;

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

        /*! @{ one mTexture object and pixel array per used mTexture */
        std::vector<cudaArray_t>         textureArrays;
        std::vector<cudaTextureObject_t> textureObjects;
        /*! @} */

        Ref<OptixDeviceContextObject> mOptixContext = nullptr;
        Ref<OptixModuleObject> mOptixModule = nullptr;
        Ref<OptixPipelineObject> mOptixPipeline = nullptr;
        Ref<OptixShaderBindingTableObject> mShaderBindingTable = nullptr;
        Ref<OptixScene> mScene = nullptr;

    };

} // namespace osc
