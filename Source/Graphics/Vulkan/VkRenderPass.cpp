#include "VkRenderPass.h"

#include <set>
#include <unordered_set>

#include "Core/Memory.h"
#include "Graphics/Vulkan/VkCoreMacros.h"

namespace SE::Graphics
{

    VkRenderPassObject::VkRenderPassObject( Ref<VkGraphicContext>                aContext,
                                                              std::vector<VkAttachmentDescription> aAttachments,
                                                              std::vector<VkSubpassDescription>    aSubpasses,
                                                              std::vector<VkSubpassDependency>     aSubpassDependencies )
        : IRenderPass{ aContext, 1 }
    {
        mVkObject = Cast<VkGraphicContext>( mGraphicContext )->CreateRenderPass( aAttachments, aSubpasses, aSubpassDependencies );
    }

    VkRenderPassObject::VkRenderPassObject( Ref<VkGraphicContext> aContext, VkFormat aFormat, uint32_t aSampleCount,
                                                              bool aIsSampled, bool aIsPresented, math::vec4 aClearColor )
        : IRenderPass{ aContext, aSampleCount }
    {
    }

    void VkRenderPassObject::CreateUnderlyingRenderpass( std::vector<VkAttachmentDescription> aAttachments,
                                                                  std::vector<VkAttachmentReference>   aColorAttachmentReferences,
                                                                  VkAttachmentReference               *aDepthAttachmentReference,
                                                                  VkAttachmentReference               *aResolveAttachmentReference )
    {
        VkSubpassDescription lSubpass{};

        lSubpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        lSubpass.colorAttachmentCount    = aColorAttachmentReferences.size();
        lSubpass.pColorAttachments       = aColorAttachmentReferences.data();
        lSubpass.pResolveAttachments     = aResolveAttachmentReference;
        lSubpass.pDepthStencilAttachment = aDepthAttachmentReference;

        mVkObject =
            Cast<VkGraphicContext>( mGraphicContext )
                ->CreateRenderPass( aAttachments, std::vector<VkSubpassDescription>{ lSubpass }, DefaultSubpassDependencies() );

        mColorAttachmentCount = aColorAttachmentReferences.size();
    }

    std::vector<VkSubpassDependency> VkRenderPassObject::DefaultSubpassDependencies()
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

    VkRenderPassObject::~VkRenderPassObject() { Cast<VkGraphicContext>( mGraphicContext )->DestroyRenderPass( mVkObject ); }

    std::vector<VkClearValue> VkRenderPassObject::GetClearValues() { return mClearValues; }

    VkAttachmentDescription VkRenderPassObject::ColorAttachment( VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled,
                                                                          bool aIsPresented, bool aIsDefined,
                                                                          VkAttachmentLoadOp  aAttachmentLoadOp,
                                                                          VkAttachmentStoreOp aAttachmentStoreOp )
    {
        VkAttachmentDescription lAttachmentSpec{};
        lAttachmentSpec.samples        = VK_SAMPLE_COUNT_VALUE( aSampleCount );
        lAttachmentSpec.format         = aFormat;
        lAttachmentSpec.loadOp         = aAttachmentLoadOp;
        lAttachmentSpec.storeOp        = aAttachmentStoreOp;
        lAttachmentSpec.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        lAttachmentSpec.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        lAttachmentSpec.initialLayout  = aIsDefined ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;

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

    VkAttachmentDescription VkRenderPassObject::DepthAttachment( bool aIsDefined, uint32_t aSampleCount,
                                                                          VkAttachmentLoadOp  aAttachmentLoadOp,
                                                                          VkAttachmentStoreOp aAttachmentStoreOp )
    {
        VkAttachmentDescription lAttachmentSpec{};
        lAttachmentSpec.samples        = VK_SAMPLE_COUNT_VALUE( aSampleCount );
        lAttachmentSpec.format         = ToVkFormat( mGraphicContext->GetDepthFormat() );
        lAttachmentSpec.loadOp         = aAttachmentLoadOp;
        lAttachmentSpec.storeOp        = aAttachmentStoreOp;
        lAttachmentSpec.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        lAttachmentSpec.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        lAttachmentSpec.initialLayout  = aIsDefined ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
        lAttachmentSpec.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        return lAttachmentSpec;
    }

} // namespace SE::Graphics