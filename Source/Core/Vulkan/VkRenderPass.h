#pragma once

#include "Core/Math/Types.h"
#include "Core/Platform/ViewportClient.h"
#include <memory>
#include <vulkan/vulkan.h>

#include "VkContext.h"
#include "VkImage.h"

#include "Core/Memory.h"

namespace LTSE::Graphics::Internal
{
    using namespace LTSE::Core;

    struct sVkRenderPassObject
    {
        VkRenderPass mVkObject    = VK_NULL_HANDLE;
        uint32_t     mSampleCount = 1;

        sVkRenderPassObject()                        = default;
        sVkRenderPassObject( sVkRenderPassObject & ) = default;
        sVkRenderPassObject( Ref<VkContext> aContext, std::vector<VkAttachmentDescription> aAttachments,
            std::vector<VkSubpassDescription> aSubpasses, std::vector<VkSubpassDependency> aSubpassDependencies );

        sVkRenderPassObject( Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented,
            math::vec4 aClearColor );

        ~sVkRenderPassObject();

        VkAttachmentDescription ColorAttachment( VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented );
        VkAttachmentDescription DepthAttachment( uint32_t aSampleCount );

        std::vector<VkClearValue> GetClearValues();

      private:
        Ref<VkContext>            mContext     = nullptr;
        std::vector<VkClearValue> mClearValues = {};
    };

} // namespace LTSE::Graphics::Internal
