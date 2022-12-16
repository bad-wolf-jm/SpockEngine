#pragma once

#include "Core/Memory.h"
#include "Graphics/Interface/IWindow.h"

#include "Core/CUDA/Texture/ColorFormat.h"

#include "Graphics/Interface/IRenderTarget.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/VkCommand.h"
#include "Graphics/Vulkan/VkCoreMacros.h"
#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkRenderTarget.h"
#include "Graphics/Vulkan/VkTexture2D.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    class ARenderTarget : public IRenderTarget
    {
      public:
        ARenderTarget() = default;
        ARenderTarget( Ref<VkGraphicContext> aGraphicContext, sRenderTargetDescription const &aRenderTargetDescription );

        ~ARenderTarget() = default;

        uint32_t GetImageCount() { return mImageCount; }

        void AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo );
        void AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                            Ref<VkTexture2D> aFramebufferImage );

        void AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat, math::vec4 aClearColor,
                            bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp, eAttachmentStoreOp eStoreOp );

        void AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat, math::vec4 aClearColor,
                            bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp, eAttachmentStoreOp eStoreOp,
                            Ref<VkTexture2D> aFramebufferImage );

        void Finalize();

        Ref<sVkAbstractRenderPassObject> GetRenderPass() { return mRenderPassObject; }

        virtual bool BeginRender();
        virtual void EndRender();
        virtual void Present();

        virtual std::vector<VkClearValue> GetClearValues();
        virtual Ref<VkRenderTarget>       GetFramebuffer();
        virtual VkSemaphore               GetImageAvailableSemaphore( uint32_t i );
        virtual VkSemaphore               GetRenderFinishedSemaphore( uint32_t i );
        virtual VkFence                   GetInFlightFence( uint32_t i );

        virtual uint32_t GetCurrentImage();

        virtual Ref<sVkCommandBufferObject> GetCurrentCommandBuffer() { return mCommandBufferObject[0]; }

        Ref<VkTexture2D> GetAttachment( std::string const &aKey );

      protected:
        Ref<sVkAbstractRenderPassObject> CreateDefaultRenderPass();

        void InitializeCommandBuffers();

        std::vector<VkClearValue>        mClearValues       = {};
        Ref<sVkAbstractRenderPassObject> mRenderPassObject  = nullptr;
        Ref<VkRenderTarget>              mFramebufferObject = nullptr;

        std::vector<Ref<sVkCommandBufferObject>> mCommandBufferObject = {};
    };
} // namespace SE::Graphics