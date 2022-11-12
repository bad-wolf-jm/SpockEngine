#include "VkLightingRenderPass.h"

#include <set>
#include <unordered_set>

#include "Core/Memory.h"
#include "VkCoreMacros.h"

namespace LTSE::Graphics::Internal
{

    sVkLightingRenderPassObject::sVkLightingRenderPassObject( Ref<VkContext> aContext,
        std::vector<VkAttachmentDescription> aAttachments, std::vector<VkSubpassDescription> aSubpasses,
        std::vector<VkSubpassDependency> aSubpassDependencies )
        : sVkAbstractRenderPassObject( aContext, aAttachments, aSubpasses, aSubpassDependencies )
    {
    }

    sVkLightingRenderPassObject::sVkLightingRenderPassObject(
        Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented, math::vec4 aClearColor )
        : sVkAbstractRenderPassObject( aContext, aFormat, aSampleCount, aIsSampled, aIsPresented, aClearColor )
    {
        VkSubpassDescription                 lSubpass{};
        std::vector<VkAttachmentDescription> lAttachments{};
        std::vector<VkAttachmentReference>   lAttachmentReferences{};

        mClearValues.resize( 5 );

        VkAttachmentDescription lColorAttachmentPosition =
            ColorAttachment( aFormat, 1, aIsSampled, aIsPresented );
        VkAttachmentReference lColorAttachmentPositionReference{};
        lColorAttachmentPositionReference.attachment = 0;
        lColorAttachmentPositionReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        lAttachments.push_back( lColorAttachmentPosition );
        lAttachmentReferences.push_back( lColorAttachmentPositionReference );
        mClearValues[0].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

        VkAttachmentDescription lDepthAttachment = DepthAttachment( 1 );
        VkAttachmentReference   lDepthAttachmentReference{};
        lDepthAttachmentReference.attachment = 1;
        lDepthAttachmentReference.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        lAttachments.push_back( lDepthAttachment );
        mClearValues[4].depthStencil = { 1.0f, 0 };

        mColorAttachmentCount = 1;

        CreateUnderlyingRenderpass( lAttachments, lAttachmentReferences, &lDepthAttachmentReference, nullptr );
    }
} // namespace LTSE::Graphics::Internal