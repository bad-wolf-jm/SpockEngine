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
        : mContext{ aContext }
    {
        mVkObject = mContext->CreateRenderPass( aAttachments, aSubpasses, aSubpassDependencies );
    }

    sVkDeferredRenderPassObject::sVkDeferredRenderPassObject(
        Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented, math::vec4 aClearColor )
        : mSampleCount{ aSampleCount }
        , mContext{ aContext }
    {
        VkSubpassDescription                 lSubpass{};
        std::vector<VkAttachmentDescription> lAttachments{};
        std::vector<VkAttachmentReference>   lAttachmentReferences{};

        mClearValues.resize( 5 );

        VkAttachmentDescription lColorAttachmentPosition = ColorAttachment( aFormat, mSampleCount, aIsSampled, aIsPresented );
        VkAttachmentReference   lColorAttachmentPositionReference{};
        lColorAttachmentPositionReference.attachment = 0;
        lColorAttachmentPositionReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        lAttachments.push_back( lColorAttachmentPosition );
        lAttachmentReferences.push_back( lColorAttachmentPositionReference );
        mClearValues[0].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

        VkAttachmentDescription lColorAttachmentNormals = ColorAttachment( aFormat, mSampleCount, aIsSampled, aIsPresented );
        VkAttachmentReference   lColorAttachmentNormalsReference{};
        lColorAttachmentNormalsReference.attachment = 1;
        lColorAttachmentNormalsReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        lAttachments.push_back( lColorAttachmentNormals );
        lAttachmentReferences.push_back( lColorAttachmentNormalsReference );
        mClearValues[1].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

        VkAttachmentDescription lColorAttachmentAlbedo = ColorAttachment( aFormat, mSampleCount, aIsSampled, aIsPresented );
        VkAttachmentReference   lColorAttachmentAlbedoReference{};
        lColorAttachmentAlbedoReference.attachment = 2;
        lColorAttachmentAlbedoReference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        lAttachments.push_back( lColorAttachmentAlbedo );
        lAttachmentReferences.push_back( lColorAttachmentAlbedoReference );
        mClearValues[2].color = { aClearColor.x, aClearColor.y, aClearColor.z, aClearColor.w };

        VkAttachmentDescription lColorAttachmentSpecular = ColorAttachment( aFormat, mSampleCount, aIsSampled, aIsPresented );
        VkAttachmentReference   lColorAttachmentSpecularReference{};
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

        lSubpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        lSubpass.colorAttachmentCount    = lAttachmentReferences.size();
        lSubpass.pColorAttachments       = lAttachmentReferences.data();
        lSubpass.pResolveAttachments     = nullptr;
        lSubpass.pDepthStencilAttachment = &lDepthAttachmentReference;

        std::vector<VkSubpassDependency> lSubpassDependencies( 2 );
        lSubpassDependencies[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
        lSubpassDependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        lSubpassDependencies[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        lSubpassDependencies[0].dstSubpass    = 0;
        lSubpassDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        lSubpassDependencies[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        lSubpassDependencies[1].srcSubpass    = 0;
        lSubpassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        lSubpassDependencies[1].srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        lSubpassDependencies[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
        lSubpassDependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        lSubpassDependencies[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

        mVkObject = mContext->CreateRenderPass( lAttachments, std::vector<VkSubpassDescription>{ lSubpass }, lSubpassDependencies );
    }

    sVkDeferredRenderPassObject::~sVkDeferredRenderPassObject() { mContext->DestroyRenderPass( mVkObject ); }

    std::vector<VkClearValue> sVkDeferredRenderPassObject::GetClearValues() { return mClearValues; }

    VkAttachmentDescription sVkDeferredRenderPassObject::ColorAttachment(
        VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented )
    {
        VkAttachmentDescription lAttachmentSpec{};
        lAttachmentSpec.samples        = VK_SAMPLE_COUNT_VALUE( aSampleCount );
        lAttachmentSpec.format         = aFormat;
        lAttachmentSpec.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        lAttachmentSpec.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        lAttachmentSpec.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        lAttachmentSpec.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        lAttachmentSpec.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;

        if( aIsSampled )
        {
            lAttachmentSpec.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        else if( aIsPresented )
        {
            lAttachmentSpec.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        }
        else
        {
            lAttachmentSpec.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        return lAttachmentSpec;
    }

    VkAttachmentDescription sVkDeferredRenderPassObject::DepthAttachment( uint32_t aSampleCount )
    {
        VkAttachmentDescription lAttachmentSpec{};
        lAttachmentSpec.samples        = VK_SAMPLE_COUNT_VALUE( aSampleCount );
        lAttachmentSpec.format         = mContext->GetDepthFormat();
        lAttachmentSpec.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        lAttachmentSpec.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        lAttachmentSpec.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        lAttachmentSpec.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        lAttachmentSpec.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        lAttachmentSpec.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        return lAttachmentSpec;
    }

} // namespace LTSE::Graphics::Internal