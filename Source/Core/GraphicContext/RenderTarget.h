#pragma once

#include "Core/Memory.h"


#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/Vulkan/VkCoreMacros.h"


namespace LTSE::Graphics
{

    using namespace LTSE::Core;

    struct RenderTargetDescription
    {
        uint32_t                           SampleCount   = 1;
        eColorFormat                       Format        = eColorFormat::RGBA8_UNORM;
        math::vec4                         ClearColor    = { 0.0f, 0.0f, 0.0f, 1.0f };
        uint32_t                           Width         = 0;
        uint32_t                           Height        = 0;
        bool                               Sampled       = false;
        bool                               Presented     = false;
        Ref<Internal::sVkFramebufferImage> OutputTexture = nullptr;
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

        Ref<Internal::sVkRenderPassObject> GetRenderPass() { return m_RenderPassObject; }

        Ref<Internal::sVkFramebufferImage> GetOutputImage() { return m_OutputTexture; }

        virtual bool                                BeginRender();
        virtual void                                EndRender();
        virtual void                                Present();
        virtual Ref<Internal::sVkFramebufferObject> GetFramebuffer();

        virtual Ref<Internal::sVkCommandBufferObject> GetCommandBuffer( uint32_t i );
        virtual VkSemaphore                           GetImageAvailableSemaphore( uint32_t i );
        virtual VkSemaphore                           GetRenderFinishedSemaphore( uint32_t i );
        virtual VkFence                               GetInFlightFence( uint32_t i );

        virtual uint32_t GetCurrentImage();

      protected:
        uint32_t mImageCount = 0;

        GraphicContext                     mGraphicContext{};
        Ref<Internal::sVkRenderPassObject> m_RenderPassObject = nullptr;
        std::vector<VkClearValue>          m_ClearColors      = {};

        Ref<Internal::sVkFramebufferImage>                 m_MSAAOutputTexture  = nullptr;
        Ref<Internal::sVkFramebufferImage>                 m_OutputTexture      = nullptr;
        Ref<Internal::sVkFramebufferImage>                 m_DepthTexture       = nullptr;
        Ref<Internal::sVkFramebufferObject>                m_FramebufferObject  = nullptr;
        std::vector<Ref<Internal::sVkCommandBufferObject>> mCommandBufferObject = {};
    };

    struct OffscreenRenderTargetDescription
    {
        uint32_t     SampleCount = 1;
        eColorFormat Format      = eColorFormat::RGBA8_UNORM;
        math::vec4   ClearColor  = { 0.0f, 0.0f, 0.0f, 1.0f };
        math::uvec2  OutputSize  = { 0, 0 };
        bool         Sampled     = false;
    };

    class OffscreenRenderTarget : public AbstractRenderTarget
    {
      public:
        OffscreenRenderTarget( GraphicContext &a_GraphicContext, OffscreenRenderTargetDescription &a_Spec );
        ~OffscreenRenderTarget() = default;

        void Resize( uint32_t aWidth, uint32_t aHeight );

        bool BeginRender();
        void EndRender();
        void Present();

        Ref<Internal::sVkFramebufferObject>   GetFramebuffer();
        Ref<Internal::sVkCommandBufferObject> GetCommandBuffer( uint32_t i );
        VkSemaphore                           GetImageAvailableSemaphore( uint32_t i );
        VkSemaphore                           GetRenderFinishedSemaphore( uint32_t i );
        VkFence                               GetInFlightFence( uint32_t i );

        uint32_t GetCurrentImage();
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

        Ref<Internal::sVkFramebufferObject>   GetFramebuffer();
        Ref<Internal::sVkCommandBufferObject> GetCommandBuffer( uint32_t i );
        VkSemaphore                           GetImageAvailableSemaphore( uint32_t i );
        VkSemaphore                           GetRenderFinishedSemaphore( uint32_t i );
        VkFence                               GetInFlightFence( uint32_t i );

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

        Ref<Internal::sVkFramebufferObject>   GetFramebuffer();
        Ref<Internal::sVkCommandBufferObject> GetCommandBuffer( uint32_t i );
        VkSemaphore                           GetImageAvailableSemaphore( uint32_t i );
        VkSemaphore                           GetRenderFinishedSemaphore( uint32_t i );
        VkFence                               GetInFlightFence( uint32_t i );

        uint32_t GetCurrentImage();

      private:
        void RecreateSwapChain();
        void CreateSwapChain();
        void CreateSyncObjects();
        void CreateRenderTargets();

        std::vector<Ref<AbstractRenderTarget>> m_RenderTargets = {};
        // Ref<Internal::VkSwapChainObject> m_SwapChainObject     = nullptr;

        VkSwapchainKHR       mVkObject      = VK_NULL_HANDLE;
        VkExtent2D           Extent         = { 0, 0 };
        VkFormat             ImageFormat    = VK_FORMAT_UNDEFINED;
        uint32_t             ImageCount     = 0;
        uint32_t             CurrentImage   = 0;
        bool                 FrameIsStarted = 0;
        std::vector<VkImage> Images         = {};

        std::vector<VkSemaphore> ImageAvailableSemaphores = {};
        std::vector<VkSemaphore> RenderFinishedSemaphores = {};
        std::vector<VkFence>     InFlightFences           = {};
    };

} // namespace LTSE::Graphics