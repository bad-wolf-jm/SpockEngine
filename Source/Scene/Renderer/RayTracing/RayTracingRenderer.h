#pragma once

#include "Core/Memory.h"

// our own classes, partly shared between host and device
#include "LaunchParams.h"

#include "Core/CUDA/Array/CudaBuffer.h"
#include "Core/Math/Types.h"

#include "Core/Optix/OptixAccelerationStructure.h"
#include "Core/Optix/OptixContext.h"
#include "Core/Optix/OptixModule.h"
#include "Core/Optix/OptixShaderBindingTable.h"

#include "Renderer/ASceneRenderer.h"

/*! \namespace osc - Optix Siggraph Course */
namespace SE::Core
{
    using namespace SE::Cuda;
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

    class RayTracingRenderer : public ASceneRenderer
    {
      public:
        RayTracingRenderer( Ref<VkGraphicContext> aGraphicContext );

        void Update( Ref<Scene> aWorld );
        void Render();
        void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

        void SetView( math::mat4 aViewMatrix );

        Ref<ITexture> GetOutputImage() { return mOutputTexture; }

        bool denoiserOn = true;
        bool accumulate = true;

      protected:
        // void computeFinalPixelColors();
        void BuildShaderBindingTable();

      protected:
        CUcontext      cudaContext;
        CUstream       stream;
        cudaDeviceProp deviceProps;

        GPUMemory mRaygenRecordsBuffer;
        GPUMemory mMissRecordsBuffer;
        GPUMemory mHitgroupRecordsBuffer;

      public:
        sLaunchParams mRayTracingParameters;

      protected:
        math::vec3 mCameraPosition{};
        math::mat3 mCameraRotation{};

      protected:
        Ref<VkGraphicContext>      mGraphicContext{};
        Ref<Graphics::VkTexture2D> mOutputTexture = nullptr;
        Ref<Graphics::VkGpuBuffer> mOutputBuffer  = nullptr;

      protected:
        GPUMemory mRayTracingParameterBuffer;

        GPUMemory fbColor;
        GPUMemory fbNormal;
        GPUMemory fbAlbedo;

        GPUMemory mDenoisedBuffer;
        GPUMemory mFinalColorBuffer;

        OptixDenoiser denoiser = nullptr;
        GPUMemory     denoiserScratch;
        GPUMemory     denoiserState;
        GPUMemory     denoiserIntensity;

        Camera lastSetCamera;

        std::vector<GPUMemory> mVertices;
        std::vector<GPUMemory> mIndices;
        std::vector<GPUMemory> mNormals;
        std::vector<GPUMemory> mTexCoords;

        Ref<OptixDeviceContextObject>      mOptixContext       = nullptr;
        Ref<OptixModuleObject>             mOptixModule        = nullptr;
        Ref<OptixPipelineObject>           mOptixPipeline      = nullptr;
        Ref<OptixShaderBindingTableObject> mShaderBindingTable = nullptr;

        Ref<Scene> mScene = nullptr;
    };

    void computeFinalPixelColors( sLaunchParams const &launchParams, Cuda::GPUMemory &denoisedBuffer,
                                  Cuda::Internal::sGPUDevicePointerView &finalColorBuffer );
} // namespace SE::Core
