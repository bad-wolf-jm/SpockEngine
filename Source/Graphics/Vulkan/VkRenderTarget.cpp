#include "VkRenderTarget.h"

#include <set>
#include <unordered_set>

namespace SE::Graphics
{
    VkRenderTarget::VkRenderTarget( Ref<VkGraphicContext> aGraphicContext, uint32_t aWidth, uint32_t aHeight, uint32_t aLayers,
                                    VkRenderPass aRenderPass, std::vector<Ref<VkTexture2D>> aAttachments )
        : mGraphicContext{ aGraphicContext }
    {
        for( auto lTextureData : aAttachments )
        {
            constexpr VkComponentMapping lSwizzles{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                                    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };

            VkImageAspectFlags lImageAspect = 0;
            if( lTextureData->mSpec.mIsDepthTexture )
                lImageAspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
            else
                lImageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;

            auto lVkImageView =
                mGraphicContext->CreateImageView( lTextureData->mVkImage, lTextureData->mSpec.mLayers, VK_IMAGE_VIEW_TYPE_2D,
                                                  ToVkFormat( lTextureData->mSpec.mFormat ), lImageAspect, lSwizzles );

            mVkImageViews.push_back( lVkImageView );
        }

        mVkFramebuffer = mGraphicContext->CreateFramebuffer( mVkImageViews, aWidth, aHeight, aLayers, aRenderPass );
    }

    VkRenderTarget::~VkRenderTarget()
    {
        for( auto &lImageView : mVkImageViews ) mGraphicContext->DestroyImageView( lImageView );
        mGraphicContext->DestroyFramebuffer( mVkFramebuffer );
    }
} // namespace SE::Graphics