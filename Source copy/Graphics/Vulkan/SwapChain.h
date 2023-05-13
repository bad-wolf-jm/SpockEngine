#pragma once

#include "VkRenderTarget.h"

#include "Graphics/Vulkan/VkCoreMacros.h"

namespace SE::Graphics
{
    class SwapChain : public VkRenderTarget
    {
      public:
        SwapChain( Ref<VkGraphicContext> aGraphicContext, Ref<IWindow> aWindow );

        bool BeginRender();
        void EndRender();
        void Present();

        std::vector<VkClearValue>           GetClearValues() { return mRenderTargets[mCurrentImage]->GetClearValues(); }
        VkFramebuffer                       GetFramebuffer() { return mRenderTargets[mCurrentImage]->GetFramebuffer(); }
        Ref<sVkAbstractRenderPassObject>    GetRenderPass() { return mRenderTargets[0]->GetRenderPass(); }
        virtual Ref<sVkCommandBufferObject> GetCurrentCommandBuffer() { return mCommandBufferObject[mCurrentImage]; }

        VkSemaphore GetImageAvailableSemaphore( uint32_t i ) { return mImageAvailableSemaphores[i]; }
        VkSemaphore GetRenderFinishedSemaphore( uint32_t i ) { return mRenderFinishedSemaphores[i]; }
        VkFence     GetInFlightFence( uint32_t i ) { return mInFlightFences[i]; }

      private:
        void RecreateSwapChain();

        Ref<IWindow> mViewportClient = nullptr;

      private:
        VkSwapchainKHR                   mVkObject                 = VK_NULL_HANDLE;
        std::vector<Ref<VkRenderTarget>> mRenderTargets            = {};
        std::vector<VkSemaphore>         mImageAvailableSemaphores = {};
        std::vector<VkSemaphore>         mRenderFinishedSemaphores = {};
        std::vector<VkFence>             mInFlightFences           = {};

        uint32_t mCurrentImage   = 0;
        bool     mFrameIsStarted = 0;
    };
} // namespace SE::Graphics