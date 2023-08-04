#pragma once

#include "VkRenderTarget.h"

#include "Graphics/Vulkan/VkCoreMacros.h"
#include "Graphics/Vulkan/VkRenderTarget.h"

#include "Graphics/Interface/ISwapChain.h"

namespace SE::Graphics
{
    class VkSwapChain : public ISwapChain
    {
      public:
        VkSwapChain( Ref<IGraphicContext> aGraphicContext, Ref<IWindow> aWindow );
        ~VkSwapChain();

        bool                            BeginRender();
        void                            EndRender();
        void                            Present();
        sRenderTargetDescription const &Spec()
        {
            return mRenderTargets[0]->mSpec;
        }

        vector_t<VkClearValue> GetClearValues()
        {
            return mRenderTargets[mCurrentImage]->GetClearValues();
        }
        VkFramebuffer GetFramebuffer()
        {
            return mRenderTargets[mCurrentImage]->GetFramebuffer();
        }
        Ref<IRenderPass> GetRenderPass()
        {
            return mRenderTargets[0]->GetRenderPass();
        }
        Ref<ICommandBuffer> GetCommandBuffer()
        {
            return mCommandBufferObject[mCurrentImage];
        }

        // Ref<sVkCommandBufferObject> GetCurrentCommandBuffer() { return mCommandBufferObject[mCurrentImage]; }

        VkSemaphore GetImageAvailableSemaphore( uint32_t i )
        {
            return mImageAvailableSemaphores[i];
        }
        VkSemaphore GetRenderFinishedSemaphore( uint32_t i )
        {
            return mRenderFinishedSemaphores[i];
        }
        VkFence GetInFlightFence( uint32_t i )
        {
            return mInFlightFences[i];
        }

      private:
        void RecreateSwapChain();

        Ref<IWindow> mViewportClient = nullptr;

      private:
        VkSurfaceKHR                          mVkSurface                = VK_NULL_HANDLE;
        VkSwapchainKHR                        mVkObject                 = VK_NULL_HANDLE;
        vector_t<Ref<VkRenderTarget>>         mRenderTargets            = {};
        vector_t<VkSemaphore>                 mImageAvailableSemaphores = {};
        vector_t<VkSemaphore>                 mRenderFinishedSemaphores = {};
        vector_t<VkFence>                     mInFlightFences           = {};
        vector_t<Ref<sVkCommandBufferObject>> mCommandBufferObject      = {};

        uint32_t mCurrentImage   = 0;
        bool     mFrameIsStarted = 0;

        void InitializeCommandBuffers();
    };
} // namespace SE::Graphics