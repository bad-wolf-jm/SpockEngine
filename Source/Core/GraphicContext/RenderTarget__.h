#pragma once

#include "Core/Memory.h"
#include "Core/Platform/ViewportClient.h"

#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/Vulkan/VkCoreMacros.h"

namespace LTSE::Graphics
{

    using namespace LTSE::Core;
    using namespace LTSE::Graphics::Internal;

    struct RenderTargetDescription
    {
        uint32_t                 SampleCount   = 1;
        eColorFormat             Format        = eColorFormat::RGBA8_UNORM;
        math::vec4               ClearColor    = { 0.0f, 0.0f, 0.0f, 1.0f };
        uint32_t                 Width         = 0;
        uint32_t                 Height        = 0;
        bool                     Sampled       = false;
        bool                     Presented     = false;
        Ref<sVkFramebufferImage> OutputTexture = nullptr;
    };

    class AbstractRenderTarget
    {
      public:
        RenderTargetDescription Spec;

        AbstractRenderTarget() = default;
        AbstractRenderTarget( GraphicContext &a_GraphicContext );
        ~AbstractRenderTarget() = default;

        uint32_t GetImageCount() { return mImageCount; }

        void Initialize( RenderTargetDescription &a_Spec );
        void InitializeCommandBuffers();

        Ref<sVkRenderPassObject> GetRenderPass() { return mRenderPassObject; }

        Ref<sVkFramebufferImage> GetOutputImage() { return mOutputTexture; }

        virtual bool                      BeginRender();
        virtual void                      EndRender();
        virtual void                      Present();
        virtual Ref<sVkFramebufferObject> GetFramebuffer();

        virtual Ref<sVkCommandBufferObject> GetCommandBuffer( uint32_t i );
        virtual VkSemaphore                 GetImageAvailableSemaphore( uint32_t i );
        virtual VkSemaphore                 GetRenderFinishedSemaphore( uint32_t i );
        virtual VkFence                     GetInFlightFence( uint32_t i );

        virtual uint32_t GetCurrentImage();

      protected:
        uint32_t mImageCount = 0;

        GraphicContext            mGraphicContext{};
        Ref<sVkRenderPassObject>  mRenderPassObject = nullptr;
        std::vector<VkClearValue> mClearColors      = {};

        Ref<sVkFramebufferImage>                 mMSAAOutputTexture   = nullptr;
        Ref<sVkFramebufferImage>                 mOutputTexture       = nullptr;
        Ref<sVkFramebufferImage>                 mDepthTexture        = nullptr;
        Ref<sVkFramebufferObject>                mFramebufferObject   = nullptr;
        std::vector<Ref<sVkCommandBufferObject>> mCommandBufferObject = {};
    };

    struct SwapChainRenderTargetDescription
    {
        uint32_t SampleCount = 1;
    };

    class SwapChainRenderTargetImage : public AbstractRenderTarget
    {
      public:
        SwapChainRenderTargetImage( GraphicContext &a_GraphicContext, RenderTargetDescription &a_Spec );
        ~SwapChainRenderTargetImage() = default;

        bool BeginRender();
        void EndRender();
        void Present();

        Ref<sVkFramebufferObject>   GetFramebuffer();
        Ref<sVkCommandBufferObject> GetCommandBuffer( uint32_t i );
        VkSemaphore                 GetImageAvailableSemaphore( uint32_t i );
        VkSemaphore                 GetRenderFinishedSemaphore( uint32_t i );
        VkFence                     GetInFlightFence( uint32_t i );

        uint32_t GetCurrentImage();
    };

    class SwapChainRenderTarget : public AbstractRenderTarget
    {
      public:
        SwapChainRenderTarget( GraphicContext &a_GraphicContext, SwapChainRenderTargetDescription &a_Spec );
        ~SwapChainRenderTarget() = default;

        bool BeginRender();
        void EndRender();
        void Present();

        Ref<sVkFramebufferObject>   GetFramebuffer();
        Ref<sVkCommandBufferObject> GetCommandBuffer( uint32_t i );
        VkSemaphore                 GetImageAvailableSemaphore( uint32_t i );
        VkSemaphore                 GetRenderFinishedSemaphore( uint32_t i );
        VkFence                     GetInFlightFence( uint32_t i );

        uint32_t GetCurrentImage();

      private:
        void RecreateSwapChain();

        std::vector<Ref<AbstractRenderTarget>> mRenderTargets = {};

        VkSwapchainKHR       mVkObject       = VK_NULL_HANDLE;
        VkExtent2D           mExtent         = { 0, 0 };
        VkFormat             mImageFormat    = VK_FORMAT_UNDEFINED;
        uint32_t             mCurrentImage   = 0;
        bool                 mFrameIsStarted = 0;
        std::vector<VkImage> mImages         = {};

        std::vector<VkSemaphore> mImageAvailableSemaphores = {};
        std::vector<VkSemaphore> mRenderFinishedSemaphores = {};
        std::vector<VkFence>     mInFlightFences           = {};
    };

} // namespace LTSE::Graphics