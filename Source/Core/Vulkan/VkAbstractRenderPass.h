#pragma once

#include "Core/GraphicContext/Window.h"
#include "Core/Math/Types.h"
#include <memory>
#include <vulkan/vulkan.h>

#include "VkContext.h"
#include "VkImage.h"

#include "Core/Memory.h"

namespace SE::Graphics::Internal
{
    using namespace SE::Core;

    struct sVkAbstractRenderPassObject
    {
        VkRenderPass mVkObject    = VK_NULL_HANDLE;
        uint32_t     mSampleCount = 1;

        sVkAbstractRenderPassObject()                                = default;
        sVkAbstractRenderPassObject( sVkAbstractRenderPassObject & ) = default;
        sVkAbstractRenderPassObject( Ref<VkContext> aContext, std::vector<VkAttachmentDescription> aAttachments,
                                     std::vector<VkSubpassDescription> aSubpasses,
                                     std::vector<VkSubpassDependency>  aSubpassDependencies );

        sVkAbstractRenderPassObject( Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled,
                                     bool aIsPresented, math::vec4 aClearColor );

        ~sVkAbstractRenderPassObject();

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
        Ref<VkContext>            mContext              = nullptr;
        std::vector<VkClearValue> mClearValues          = {};
        uint32_t                  mColorAttachmentCount = 0;
    };

} // namespace SE::Graphics::Internal
