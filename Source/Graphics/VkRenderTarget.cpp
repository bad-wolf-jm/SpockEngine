#include "VkRenderTarget.h"

#include <set>
#include <unordered_set>

#include "VkCoreMacros.h"

namespace SE::Graphics
{
    VkRenderTarget::VkRenderTarget( Ref<VkContext> aContext, uint32_t aWidth, uint32_t aHeight, uint32_t aLayers,
                                    VkRenderPass aRenderPass, std::vector<Ref<VkTexture2D>> aAttachments )
        : mContext{ aContext }
    {

        std::vector<VkImageView> lAttachmentViews( aAttachments.size() );

        for( uint32_t i = 0; i < aAttachments.size(); i++ ) lAttachmentViews[i] = aAttachments[i]->mImageView->mVkObject;

        mVkObject = mContext->CreateFramebuffer( lAttachmentViews, aWidth, aHeight, aLayers, aRenderPass );
    }

    VkRenderTarget::~VkRenderTarget() { mContext->DestroyFramebuffer( mVkObject ); }
} // namespace SE::Graphics