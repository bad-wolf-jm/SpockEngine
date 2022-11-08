#include "VkRenderPass.h"

#include <set>
#include <unordered_set>

#include "Core/Memory.h"
#include "VkCoreMacros.h"

namespace LTSE::Graphics::Internal
{

    sVkAbstractRenderPassObject::sVkAbstractRenderPassObject( Ref<VkContext> aContext,
        std::vector<VkAttachmentDescription> aAttachments, std::vector<VkSubpassDescription> aSubpasses,
        std::vector<VkSubpassDependency> aSubpassDependencies )
        : mContext{ aContext }
    {
        mVkObject = mContext->CreateRenderPass( aAttachments, aSubpasses, aSubpassDependencies );
    }

    sVkAbstractRenderPassObject::sVkAbstractRenderPassObject(
        Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented, math::vec4 aClearColor )
        : mSampleCount{ aSampleCount }
        , mContext{ aContext }
    {
    }

    void sVkAbstractRenderPassObject::CreateUnderlyingRenderpass( std::vector<VkAttachmentDescription> aAttachments,
        std::vector<VkAttachmentReference> aColorAttachmentReferences, VkAttachmentReference *aDepthAttachmentReference,
        VkAttachmentReference *aResolveAttachmentReference )
    {
        VkSubpassDescription lSubpass{};

        lSubpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        lSubpass.colorAttachmentCount    = aColorAttachmentReferences.size();
        lSubpass.pColorAttachments       = aColorAttachmentReferences.data();
        lSubpass.pResolveAttachments     = aResolveAttachmentReference;
        lSubpass.pDepthStencilAttachment = aDepthAttachmentReference;

        mVkObject =
            mContext->CreateRenderPass( aAttachments, std::vector<VkSubpassDescription>{ lSubpass }, DefaultSubpassDependencies() );
    }

    std::vector<VkSubpassDependency> sVkAbstractRenderPassObject::DefaultSubpassDependencies()
    {
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

        return lSubpassDependencies;
    }

    sVkAbstractRenderPassObject::~sVkAbstractRenderPassObject() { mContext->DestroyRenderPass( mVkObject ); }

    std::vector<VkClearValue> sVkAbstractRenderPassObject::GetClearValues() { return mClearValues; }

    VkAttachmentDescription sVkAbstractRenderPassObject::ColorAttachment(
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

    VkAttachmentDescription sVkAbstractRenderPassObject::DepthAttachment( uint32_t aSampleCount )
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