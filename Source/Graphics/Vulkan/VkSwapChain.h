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
        VkSwapChain( ref_t<IGraphicContext> aGraphicContext, ref_t<IWindow> aWindow );
        ~VkSwapChain();

        bool                            BeginRender();
        void                            EndRender();
        void                            Present();
        sRenderTargetDescription const &Spec()
        {
            return mRenderTargets[0]->mSpec;
        }

        vec_t<VkClearValue> GetClearValues()
        {
            return mRenderTargets[mCurrentImage]->GetClearValues();
        }
        VkFramebuffer GetFramebuffer()
        {
            return mRenderTargets[mCurrentImage]->GetFramebuffer();
        }
        ref_t<IRenderPass> GetRenderPass()
        {
            return mRenderTargets[0]->GetRenderPass();
        }
        ref_t<ICommandBuffer> GetCommandBuffer()
        {
            return mCommandBufferObject[mCurrentImage];
        }

        // ref_t<sVkCommandBufferObject> GetCurrentCommandBuffer() { return mCommandBufferObject[mCurrentImage]; }

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

        ref_t<IWindow> mViewportClient = nullptr;

      private:
        VkSurfaceKHR                             mVkSurface                = VK_NULL_HANDLE;
        VkSwapchainKHR                           mVkObject                 = VK_NULL_HANDLE;
        vec_t<ref_t<VkRenderTarget>>         mRenderTargets            = {};
        vec_t<VkSemaphore>                 mImageAvailableSemaphores = {};
        vec_t<VkSemaphore>                 mRenderFinishedSemaphores = {};
        vec_t<VkFence>                     mInFlightFences           = {};
        vec_t<ref_t<sVkCommandBufferObject>> mCommandBufferObject      = {};

        uint32_t mCurrentImage   = 0;
        bool     mFrameIsStarted = 0;

        void InitializeCommandBuffers();
    };
} // namespace SE::Graphics