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

    struct sVkAbstractRenderPassObject
    {
        VkRenderPass mVkObject    = VK_NULL_HANDLE;
        uint32_t     mSampleCount = 1;

        sVkAbstractRenderPassObject()                                = default;
        sVkAbstractRenderPassObject( sVkAbstractRenderPassObject & ) = default;
        sVkAbstractRenderPassObject( Ref<VkContext> aContext, std::vector<VkAttachmentDescription> aAttachments,
            std::vector<VkSubpassDescription> aSubpasses, std::vector<VkSubpassDependency> aSubpassDependencies );

        sVkAbstractRenderPassObject( Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled,
            bool aIsPresented, math::vec4 aClearColor );

        ~sVkAbstractRenderPassObject();

        VkAttachmentDescription ColorAttachment( VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented );
        VkAttachmentDescription DepthAttachment( uint32_t aSampleCount );

        std::vector<VkClearValue> GetClearValues();

        std::vector<VkSubpassDependency> DefaultSubpassDependencies();

        void CreateUnderlyingRenderpass(
            std::vector<VkAttachmentDescription> aAttachments, std::vector<VkAttachmentReference> aAttachmentReferences ,
            VkAttachmentReference *aDepthAttachmentReference, VkAttachmentReference *aResolveAttachmentReference);

        uint32_t GetColorAttachmentCount() { return mColorAttachmentCount; }

      protected:
        Ref<VkContext>            mContext     = nullptr;
        std::vector<VkClearValue> mClearValues = {};
        uint32_t mColorAttachmentCount = 0;
    };

} // namespace LTSE::Graphics::Internal
