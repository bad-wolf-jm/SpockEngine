#include "VkRenderPass.h"

#include <set>
#include <unordered_set>

#include "Core/Memory.h"
#include "VkCoreMacros.h"

namespace LTSE::Graphics::Internal
{

    sVkRenderPassObject::sVkRenderPassObject( Ref<VkContext> aContext, std::vector<VkAttachmentDescription> aAttachments,
        std::vector<VkSubpassDescription> aSubpasses, std::vector<VkSubpassDependency> aSubpassDependencies )
        : mContext{ aContext }
    {
        mVkObject = mContext->CreateRenderPass( aAttachments, aSubpasses, aSubpassDependencies );
    }

    sVkRenderPassObject::sVkRenderPassObject(
        Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented, math::vec4 aClearColor )
        : mSampleCount{ aSampleCount }
        , mContext{ aContext }
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

            lSubpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
            lSubpass.colorAttachmentCount    = 1;
            lSubpass.pColorAttachments       = &lColorAttachmentReference;
            lSubpass.pResolveAttachments     = nullptr;
            lSubpass.pDepthStencilAttachment = &lDepthAttachmentReference;
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

            lSubpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
            lSubpass.colorAttachmentCount    = 1;
            lSubpass.pColorAttachments       = &lMSAAColorAttachmentReference;
            lSubpass.pResolveAttachments     = &lResolveColorAttachmentReference;
            lSubpass.pDepthStencilAttachment = &lDepthAttachmentReference;
        }

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

    sVkRenderPassObject::~sVkRenderPassObject() { mContext->DestroyRenderPass( mVkObject ); }

    std::vector<VkClearValue> sVkRenderPassObject::GetClearValues() { return mClearValues; }

    VkAttachmentDescription sVkRenderPassObject::ColorAttachment(
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

    VkAttachmentDescription sVkRenderPassObject::DepthAttachment( uint32_t aSampleCount )
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