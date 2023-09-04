#pragma once

#include "Core/Memory.h"
#include "Graphics/Interface/IWindow.h"

#include "Core/CUDA/Texture/ColorFormat.h"

#include "Graphics/Interface/IRenderTarget.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/VkCommand.h"
#include "Graphics/Vulkan/VkCoreMacros.h"
#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkTexture2D.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    class VkRenderTarget : public IRenderTarget
    {
      public:
        VkRenderTarget() = default;
        VkRenderTarget( ref_t<VkGraphicContext> aGraphicContext, sRenderTargetDescription const &aRenderTargetDescription );

        ~VkRenderTarget();

        void AddAttachment( string_t const &aAttachmentID, sAttachmentDescription const &aCreateInfo );

        void AddAttachment( string_t const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                            ref_t<VkTexture2D> aFramebufferImage );

        void AddAttachment( string_t const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                            ref_t<VkTexture2D> aFramebufferImage, eCubeFace aFace );

        void AddAttachment( string_t const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat, math::vec4 aClearColor,
                            bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp, eAttachmentStoreOp eStoreOp );

        void AddAttachment( string_t const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat, math::vec4 aClearColor,
                            bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp, eAttachmentStoreOp eStoreOp,
                            ref_t<VkTexture2D> aFramebufferImage );

        void Finalize();

        ref_t<IRenderPass> GetRenderPass()
        {
            return mRenderPassObject;
        }
        ref_t<ICommandBuffer> GetCommandBuffer()
        {
            return mCommandBufferObject[0];
        }

        virtual bool BeginRender();
        virtual void EndRender();
        virtual void Present();

        virtual vector_t<VkClearValue> GetClearValues();
        virtual VkFramebuffer             GetFramebuffer();
        virtual VkSemaphore               GetImageAvailableSemaphore( uint32_t i );
        virtual VkSemaphore               GetRenderFinishedSemaphore( uint32_t i );
        virtual VkFence                   GetInFlightFence( uint32_t i );

        virtual uint32_t GetCurrentImage();

        // virtual ref_t<sVkCommandBufferObject> GetCurrentCommandBuffer() { return mCommandBufferObject[0]; }

        ref_t<VkTexture2D> GetAttachment( string_t const &aKey );

      protected:
        ref_t<VkRenderPassObject> CreateDefaultRenderPass();

        void InitializeCommandBuffers();

        vector_t<VkClearValue> mClearValues      = {};
        ref_t<VkRenderPassObject>   mRenderPassObject = nullptr;

        vector_t<ref_t<sVkCommandBufferObject>> mCommandBufferObject = {};

        VkFramebuffer            mVkFramebuffer = VK_NULL_HANDLE;
        vector_t<VkImageView> mVkImageViews  = {};
    };
} // namespace SE::Graphics