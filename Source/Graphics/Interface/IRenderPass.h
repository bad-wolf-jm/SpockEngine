#pragma once

#include "Core/Memory.h"
#include "IGraphicContext.h"
#include "IRenderTarget.h"

#include <vector>
#include <vulkan/vulkan.h>

namespace SE::Graphics
{
    class IRenderPass
    {
        VkRenderPass mVkObject    = VK_NULL_HANDLE;
        uint32_t     mSampleCount = 1;

        IRenderPass()                = default;
        IRenderPass( IRenderPass & ) = default;
        IRenderPass( Ref<IGraphicContext> aContext, std::vector<VkAttachmentDescription> aAttachments,
                     std::vector<VkSubpassDescription> aSubpasses, std::vector<VkSubpassDependency> aSubpassDependencies );

        IRenderPass( Ref<VkGraphicContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented,
                     math::vec4 aClearColor );

        ~IRenderPass();

        VkAttachmentDescription ColorAttachment( VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented,
                                                 bool aIsDefined, VkAttachmentLoadOp aAttachmentLoadOp,
                                                 VkAttachmentStoreOp aAttachmentStoreOp );
        VkAttachmentDescription DepthAttachment( bool aIsDefined, uint32_t aSampleCount, VkAttachmentLoadOp aAttachmentLoadOp,
                                                 VkAttachmentStoreOp aAttachmentStoreOp );

        std::vector<VkClearValue> GetClearValues();

        std::vector<VkSubpassDependency> DefaultSubpassDependencies();

        void CreateUnderlyingRenderpass( std::vector<VkAttachmentDescription> aAttachments,
                                         std::vector<VkAttachmentReference>   aAttachmentReferences,
                                         VkAttachmentReference               *aDepthAttachmentReference,
                                         VkAttachmentReference               *aResolveAttachmentReference );

        uint32_t GetColorAttachmentCount() { return mColorAttachmentCount; }

      protected:
        Ref<VkGraphicContext>     mContext              = nullptr;
        std::vector<VkClearValue> mClearValues          = {};
        uint32_t                  mColorAttachmentCount = 0;
    };
} // namespace SE::Graphics