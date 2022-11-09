#include "VkRenderPass.h"

#include <set>
#include <unordered_set>

#include "Core/Memory.h"
#include "VkCoreMacros.h"

namespace LTSE::Graphics::Internal
{

    sVkRenderPassObject::sVkRenderPassObject( Ref<VkContext> aContext, std::vector<VkAttachmentDescription> aAttachments,
        std::vector<VkSubpassDescription> aSubpasses, std::vector<VkSubpassDependency> aSubpassDependencies )
        : sVkAbstractRenderPassObject( aContext, aAttachments, aSubpasses, aSubpassDependencies )
    {
    }

    sVkRenderPassObject::sVkRenderPassObject(
        Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented, math::vec4 aClearColor )
        : sVkAbstractRenderPassObject( aContext, aFormat, aSampleCount, aIsSampled, aIsPresented, aClearColor )
    {
        VkSubpassDescription                 lSubpass{};
        std::vector<VkAttachmentDescription> lAttachments{};

        if( mSampleCount == 1 )
        {
            lAttachments.resize( 2 );
            mClearValues.resize( 2 );
            VkAttachmentDescription lColorAttachment = ColorAttachment( aFormat, mSampleCount, aIsSampled, aIsPresented );
            VkAttachmentReference   lColorAttachmentReference{};
            lColorAttachmentReference.attachment = 0;
            lColorAttachmentReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentDescription lDepthAttachment = DepthAttachment( 1 );
            VkAttachmentReference   lDepthAttachmentReference{};
            lDepthAttachmentReference.attachment = 1;
            lDepthAttachmentReference.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            lAttachments[0] = lColorAttachment;
            lAttachments[1] = lDepthAttachment;

            mClearValues[0]       = VkClearValue{};
            mClearValues[0].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

            mClearValues[1]       = VkClearValue{};
            mClearValues[1].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

            mColorAttachmentCount = 1;

            CreateUnderlyingRenderpass(
                lAttachments, std::vector<VkAttachmentReference>{ lColorAttachmentReference }, &lDepthAttachmentReference, nullptr );
        }
        else
        {
            lAttachments.resize( 3 );
            mClearValues.resize( 3 );

            VkAttachmentDescription lMSAAColorAttachment = ColorAttachment( aFormat, mSampleCount, false, false );
            VkAttachmentReference   lMSAAColorAttachmentReference{};
            lMSAAColorAttachmentReference.attachment = 0;
            lMSAAColorAttachmentReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentDescription lResolveColorAttachment = ColorAttachment( aFormat, 1, aIsSampled, aIsPresented );
            VkAttachmentReference   lResolveColorAttachmentReference{};
            lResolveColorAttachmentReference.attachment = 1;
            lResolveColorAttachmentReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentDescription lDepthAttachment = DepthAttachment( mSampleCount );
            VkAttachmentReference   lDepthAttachmentReference{};
            lDepthAttachmentReference.attachment = 2;
            lDepthAttachmentReference.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            lAttachments[0] = lMSAAColorAttachment;
            lAttachments[1] = lResolveColorAttachment;
            lAttachments[2] = lDepthAttachment;

            mClearValues[0]       = VkClearValue{};
            mClearValues[0].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

            mClearValues[1]       = VkClearValue{};
            mClearValues[1].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

            mClearValues[2]              = VkClearValue{};
            mClearValues[2].depthStencil = { 1.0f, 0 };

            mColorAttachmentCount = 1;

            CreateUnderlyingRenderpass(
                lAttachments, std::vector<VkAttachmentReference>{ lMSAAColorAttachmentReference }, &lDepthAttachmentReference, &lResolveColorAttachmentReference );
        }
    }
} // namespace LTSE::Graphics::Internal