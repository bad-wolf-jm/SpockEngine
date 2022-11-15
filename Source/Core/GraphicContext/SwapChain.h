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

        std::vector<VkClearValue> GetClearValues()
        {
            return mRenderTargets[mCurrentImage]->GetClearValues();
        }

        Ref<sVkFramebufferObject>        GetFramebuffer() { return mRenderTargets[mCurrentImage]->GetFramebuffer(); }
        Ref<sVkAbstractRenderPassObject> GetRenderPass() { return mRenderTargets[0]->GetRenderPass(); }

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