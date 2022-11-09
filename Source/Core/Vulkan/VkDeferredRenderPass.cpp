#include "VkDeferredRenderPass.h"

#include <set>
#include <unordered_set>

#include "Core/Memory.h"
#include "VkCoreMacros.h"

namespace LTSE::Graphics::Internal
{

    sVkDeferredRenderPassObject::sVkDeferredRenderPassObject( Ref<VkContext> aContext,
        std::vector<VkAttachmentDescription> aAttachments, std::vector<VkSubpassDescription> aSubpasses,
        std::vector<VkSubpassDependency> aSubpassDependencies )
        : sVkAbstractRenderPassObject( aContext, aAttachments, aSubpasses, aSubpassDependencies )
    {
    }

    sVkDeferredRenderPassObject::sVkDeferredRenderPassObject(
        Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented, math::vec4 aClearColor )
        : sVkAbstractRenderPassObject( aContext, aFormat, aSampleCount, aIsSampled, aIsPresented, aClearColor )
    {
        VkSubpassDescription                 lSubpass{};
        std::vector<VkAttachmentDescription> lAttachments{};
        std::vector<VkAttachmentReference>   lAttachmentReferences{};

        mClearValues.resize( 5 );

        VkAttachmentDescription lColorAttachmentPosition =
            ColorAttachment( VK_FORMAT_R16G16B16A16_SFLOAT, mSampleCount, aIsSampled, aIsPresented );
        VkAttachmentReference lColorAttachmentPositionReference{};
        lColorAttachmentPositionReference.attachment = 0;
        lColorAttachmentPositionReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        lAttachments.push_back( lColorAttachmentPosition );
        lAttachmentReferences.push_back( lColorAttachmentPositionReference );
        mClearValues[0].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

        VkAttachmentDescription lColorAttachmentNormals =
            ColorAttachment( VK_FORMAT_R16G16B16A16_SFLOAT, mSampleCount, aIsSampled, aIsPresented );
        VkAttachmentReference lColorAttachmentNormalsReference{};
        lColorAttachmentNormalsReference.attachment = 1;
        lColorAttachmentNormalsReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        lAttachments.push_back( lColorAttachmentNormals );
        lAttachmentReferences.push_back( lColorAttachmentNormalsReference );
        mClearValues[1].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

        VkAttachmentDescription lColorAttachmentAlbedo =
            ColorAttachment( VK_FORMAT_R8G8B8A8_UNORM, mSampleCount, aIsSampled, aIsPresented );
        VkAttachmentReference lColorAttachmentAlbedoReference{};
        lColorAttachmentAlbedoReference.attachment = 2;
        lColorAttachmentAlbedoReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        lAttachments.push_back( lColorAttachmentAlbedo );
        lAttachmentReferences.push_back( lColorAttachmentAlbedoReference );
        mClearValues[2].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

        VkAttachmentDescription lColorAttachmentSpecular =
            ColorAttachment( VK_FORMAT_R8G8B8A8_UNORM, mSampleCount, aIsSampled, aIsPresented );
        VkAttachmentReference lColorAttachmentSpecularReference{};
        lColorAttachmentSpecularReference.attachment = 3;
        lColorAttachmentSpecularReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        lAttachments.push_back( lColorAttachmentSpecular );
        lAttachmentReferences.push_back( lColorAttachmentSpecularReference );
        mClearValues[3].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

        VkAttachmentDescription lDepthAttachment = DepthAttachment( mSampleCount );
        VkAttachmentReference   lDepthAttachmentReference{};
        lDepthAttachmentReference.attachment = 4;
        lDepthAttachmentReference.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        lAttachments.push_back( lDepthAttachment );
        mClearValues[4].depthStencil = { 1.0f, 0 };

        mColorAttachmentCount = 4;

        CreateUnderlyingRenderpass( lAttachments, lAttachmentReferences, &lDepthAttachmentReference, nullptr );
    }
} // namespace LTSE::Graphics::Internal