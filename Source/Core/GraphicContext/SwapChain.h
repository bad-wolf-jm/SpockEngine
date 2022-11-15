#pragma once

#include "ARenderTarget.h"

#include "Core/Vulkan/VkCoreMacros.h"

namespace LTSE::Graphics
{
    class SwapChain : public ARenderTarget
    {
      public:
        SwapChain( GraphicContext &aGraphicContext );

        bool BeginRender();
        void EndRender();
        void Present();

        std::vector<VkClearValue>           GetClearValues() { return mRenderTargets[mCurrentImage]->GetClearValues(); }
        Ref<sVkFramebufferObject>           GetFramebuffer() { return mRenderTargets[mCurrentImage]->GetFramebuffer(); }
        Ref<sVkAbstractRenderPassObject>    GetRenderPass() { return mRenderTargets[0]->GetRenderPass(); }
        virtual Ref<sVkCommandBufferObject> GetCurrentCommandBuffer() { return mCommandBufferObject[mCurrentImage]; }

        VkSemaphore GetImageAvailableSemaphore( uint32_t i ) { return mImageAvailableSemaphores[i]; }
        VkSemaphore GetRenderFinishedSemaphore( uint32_t i ) { return mRenderFinishedSemaphores[i]; }
        VkFence     GetInFlightFence( uint32_t i ) { return mInFlightFences[i]; }

      private:
        void RecreateSwapChain();

      private:
        VkSwapchainKHR                  mVkObject                 = VK_NULL_HANDLE;
        std::vector<Ref<ARenderTarget>> mRenderTargets            = {};
        std::vector<VkSemaphore>        mImageAvailableSemaphores = {};
        std::vector<VkSemaphore>        mRenderFinishedSemaphores = {};
        std::vector<VkFence>            mInFlightFences           = {};

        uint32_t mCurrentImage   = 0;
        bool     mFrameIsStarted = 0;
    };
} // namespace LTSE::Graphics