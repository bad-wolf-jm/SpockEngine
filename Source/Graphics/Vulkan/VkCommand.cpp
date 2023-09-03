#include "VkCommand.h"

#include <set>
#include <unordered_set>

#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkCoreMacros.h"

namespace SE::Graphics
{

    sVkCommandBufferObject::sVkCommandBufferObject( ref_t<VkGraphicContext> aContext, VkCommandBuffer aCommandBuffer )
        : ICommandBuffer( aContext )
        , mVkObject{ aCommandBuffer }
    {
    }

    sVkCommandBufferObject::sVkCommandBufferObject( ref_t<VkGraphicContext> aContext )
        : ICommandBuffer( aContext )
    {
        mVkObject    = Cast<VkGraphicContext>( mGraphicContext )->AllocateCommandBuffer( 1 )[0];
        mSubmitFence = Cast<VkGraphicContext>( mGraphicContext )->CreateFence();
    }

    sVkCommandBufferObject::~sVkCommandBufferObject()
    {
        Cast<VkGraphicContext>( mGraphicContext )->DestroyCommandBuffer( mVkObject );
        Cast<VkGraphicContext>( mGraphicContext )->DestroyFence( mSubmitFence );
    }

    void sVkCommandBufferObject::Begin() { Begin( 0 ); }

    void sVkCommandBufferObject::Begin( VkCommandBufferUsageFlags aUsage )
    {
        Cast<VkGraphicContext>( mGraphicContext )->WaitForFence( mSubmitFence );

        VkCommandBufferBeginInfo lCommandBufferBeginInfo{};
        lCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        lCommandBufferBeginInfo.flags = aUsage;

        VK_CHECK_RESULT( vkBeginCommandBuffer( mVkObject, &lCommandBufferBeginInfo ) );
    }

    void sVkCommandBufferObject::BeginRenderPass( ref_t<IRenderPass> aRenderPass, VkFramebuffer aFrameBuffer, math::uvec2 aExtent,
                                                  vec_t<VkClearValue> aClearValues )
    {
        VkRenderPassBeginInfo lRenderPassInfo{};
        lRenderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        lRenderPassInfo.renderPass        = Cast<VkRenderPassObject>( aRenderPass )->mVkObject;
        lRenderPassInfo.framebuffer       = aFrameBuffer;
        lRenderPassInfo.renderArea.offset = { 0, 0 };
        lRenderPassInfo.renderArea.extent = { aExtent.x, aExtent.y };
        lRenderPassInfo.clearValueCount   = static_cast<uint32_t>( aClearValues.size() );
        lRenderPassInfo.pClearValues      = aClearValues.data();

        vkCmdBeginRenderPass( mVkObject, &lRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE );
    }

    void sVkCommandBufferObject::EndRenderPass() { vkCmdEndRenderPass( mVkObject ); }

    void sVkCommandBufferObject::SetViewport( math::ivec2 aOffset, math::uvec2 aSize )
    {
        VkViewport lViewportDescription;
        lViewportDescription.x        = aOffset.x;
        lViewportDescription.y        = aOffset.y;
        lViewportDescription.width    = aSize.x;
        lViewportDescription.height   = aSize.y;
        lViewportDescription.minDepth = 0.0f;
        lViewportDescription.maxDepth = 1.0f;
        vkCmdSetViewport( mVkObject, 0, 1, &lViewportDescription );
    }

    void sVkCommandBufferObject::SetScissor( math::ivec2 aOffset, math::uvec2 aSize )
    {
        VkRect2D lScissorDescription;
        lScissorDescription.offset.x      = (int32_t)( aOffset.x );
        lScissorDescription.offset.y      = (int32_t)( aOffset.y );
        lScissorDescription.extent.width  = ( aSize.x );
        lScissorDescription.extent.height = ( aSize.y );
        vkCmdSetScissor( mVkObject, 0, 1, &lScissorDescription );
    }

    void sVkCommandBufferObject::Draw( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset,
                                       uint32_t aInstanceCount, uint32_t aFirstInstance )
    {
        vkCmdDraw( mVkObject, aVertexCount, aInstanceCount, aVertexOffset, aFirstInstance );
    }

    void sVkCommandBufferObject::DrawIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset,
                                              uint32_t aInstanceCount, uint32_t aFirstInstance )
    {
        vkCmdDrawIndexed( mVkObject, aVertexCount, aInstanceCount, aVertexOffset, aVertexBufferOffset, aFirstInstance );
    }

    void sVkCommandBufferObject::Bind( ref_t<sVkPipelineObject> aPipeline, VkPipelineBindPoint aBindPoint )
    {
        vkCmdBindPipeline( mVkObject, aBindPoint, aPipeline->mVkObject );
    }

    void sVkCommandBufferObject::Bind( VkBuffer aVertexBuffer, uint32_t aBindPoint )
    {
        VkDeviceSize lOffsets[] = { 0 };
        vkCmdBindVertexBuffers( mVkObject, aBindPoint, 1, &aVertexBuffer, lOffsets );
    }

    void sVkCommandBufferObject::Bind( VkBuffer aVertexBuffer, VkBuffer a_IndexBuffer, uint32_t aBindPoint )
    {
        VkDeviceSize lOffsets[] = { 0 };
        vkCmdBindVertexBuffers( mVkObject, aBindPoint, 1, &aVertexBuffer, lOffsets );
        vkCmdBindIndexBuffer( mVkObject, a_IndexBuffer, 0, VK_INDEX_TYPE_UINT32 );
    }

    void sVkCommandBufferObject::Bind( ref_t<sVkDescriptorSetObject> aDescriptorSet, VkPipelineBindPoint aBindPoint,
                                       ref_t<sVkPipelineLayoutObject> aPipelineLayout, uint32_t aSetIndex, int32_t aDynamicOffset )
    {
        VkDescriptorSet lDescriptorSetArray[1] = { aDescriptorSet->mVkObject };
        if( aDynamicOffset == -1 )
        {
            vkCmdBindDescriptorSets( mVkObject, aBindPoint, aPipelineLayout->mVkObject, aSetIndex, 1, lDescriptorSetArray, 0,
                                     nullptr );
        }
        else
        {
            const uint32_t l_DynOffset = static_cast<uint32_t>( aDynamicOffset );
            vkCmdBindDescriptorSets( mVkObject, aBindPoint, aPipelineLayout->mVkObject, aSetIndex, 1, lDescriptorSetArray, 1,
                                     &l_DynOffset );
        }
    }

    void sVkCommandBufferObject::Bind( void *aDescriptorSet, VkPipelineBindPoint aBindPoint,
                                       ref_t<sVkPipelineLayoutObject> aPipelineLayout, uint32_t aSetIndex, int32_t aDynamicOffset )
    {
        VkDescriptorSet lDescriptorSetArray[1] = { (VkDescriptorSet)aDescriptorSet };
        if( aDynamicOffset == -1 )
        {
            vkCmdBindDescriptorSets( mVkObject, aBindPoint, aPipelineLayout->mVkObject, aSetIndex, 1, lDescriptorSetArray, 0,
                                     nullptr );
        }
        else
        {
            const uint32_t l_DynOffset = static_cast<uint32_t>( aDynamicOffset );
            vkCmdBindDescriptorSets( mVkObject, aBindPoint, aPipelineLayout->mVkObject, aSetIndex, 1, lDescriptorSetArray, 1,
                                     &l_DynOffset );
        }
    }

    void sVkCommandBufferObject::ImageMemoryBarrier( VkImage aImage, VkImageLayout aOldLayout, VkImageLayout aNewLayout,
                                                     uint32_t aMipCount, uint32_t aLayerCount )
    {
        VkImageMemoryBarrier lBarrier{};
        lBarrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        lBarrier.oldLayout                       = aOldLayout;
        lBarrier.newLayout                       = aNewLayout;
        lBarrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        lBarrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        lBarrier.image                           = aImage;
        lBarrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        lBarrier.subresourceRange.baseMipLevel   = 0;
        lBarrier.subresourceRange.levelCount     = aMipCount;
        lBarrier.subresourceRange.baseArrayLayer = 0;
        lBarrier.subresourceRange.layerCount     = aLayerCount;

        VkPipelineStageFlags lSourceStage;
        VkPipelineStageFlags lDestinationStage;

        if( ( aOldLayout == VK_IMAGE_LAYOUT_UNDEFINED ) && ( aNewLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ) )
        {
            lBarrier.srcAccessMask = 0;
            lBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            lSourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            lDestinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if( ( aOldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ) && ( aNewLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ) )
        {
            lBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            lBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            lSourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
            lDestinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if( ( aOldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ) && ( aNewLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ) )
        {
            lBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            lBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

            lSourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
            lDestinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if( ( aOldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ) && ( aNewLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ) )
        {
            lBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            lBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

            lSourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
            lDestinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else
        {
            lSourceStage      = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            lDestinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }

        vkCmdPipelineBarrier( mVkObject, lSourceStage, lDestinationStage, 0, 0, nullptr, 0, nullptr, 1, &lBarrier );
    }

    void sVkCommandBufferObject::CopyBuffer( VkBuffer aSource, uint32_t aSourceOffset, uint32_t aSize, VkBuffer aDest,
                                             uint32_t aDestOffset )
    {
        VkBufferCopy lCopyRegion{};
        lCopyRegion.srcOffset = aSourceOffset;
        lCopyRegion.dstOffset = aDestOffset;
        lCopyRegion.size      = aSize;
        vkCmdCopyBuffer( mVkObject, aSource, aDest, 1, &lCopyRegion );
    }

    void sVkCommandBufferObject::CopyBuffer( VkBuffer aSource, VkImage aDestination, sImageRegion const &aBufferRegion,
                                             sImageRegion const &aImageRegion )
    {
        CopyBuffer( aSource, aDestination, vec_t<sImageRegion>{ aBufferRegion }, aImageRegion );
    }

    void sVkCommandBufferObject::CopyBuffer( VkBuffer aSource, VkImage aDestination, vec_t<sImageRegion> aBufferRegions,
                                             sImageRegion const &aImageRegion )
    {
        vec_t<VkBufferImageCopy> lBufferCopyRegions;

        for( auto const &lRegion : aBufferRegions )
        {
            VkBufferImageCopy lBufferCopyRegion{};

            lBufferCopyRegion.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            lBufferCopyRegion.imageSubresource.mipLevel       = lRegion.mBaseMipLevel;
            lBufferCopyRegion.imageSubresource.baseArrayLayer = lRegion.mBaseLayer;
            lBufferCopyRegion.imageSubresource.layerCount     = lRegion.mLayerCount;
            lBufferCopyRegion.imageExtent.width               = lRegion.mWidth;
            lBufferCopyRegion.imageExtent.height              = lRegion.mHeight;
            lBufferCopyRegion.imageExtent.depth               = lRegion.mDepth;
            lBufferCopyRegion.bufferOffset                    = lRegion.mOffset;

            lBufferCopyRegions.push_back( lBufferCopyRegion );
        }

        VkImageSubresourceRange lSubresourceRange{};
        lSubresourceRange.aspectMask   = VK_IMAGE_ASPECT_COLOR_BIT;
        lSubresourceRange.baseMipLevel = aImageRegion.mBaseMipLevel;
        lSubresourceRange.levelCount   = aImageRegion.mMipLevelCount;
        lSubresourceRange.layerCount   = aImageRegion.mLayerCount;

        ImageMemoryBarrier( aDestination, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, aImageRegion.mMipLevelCount,
                            aImageRegion.mLayerCount );

        vkCmdCopyBufferToImage( mVkObject, aSource, aDestination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                static_cast<uint32_t>( lBufferCopyRegions.size() ), lBufferCopyRegions.data() );
    }

    void sVkCommandBufferObject::CopyImage( VkImage aSource, sImageRegion const &aSourceRegion, VkImage aDestination,
                                            sImageRegion const &aDestRegion )
    {
        ImageMemoryBarrier( aDestination, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 1 );

        VkImageCopy lCopyRegion{};

        lCopyRegion.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        lCopyRegion.srcSubresource.baseArrayLayer = aSourceRegion.mBaseLayer;
        lCopyRegion.srcSubresource.mipLevel       = aSourceRegion.mBaseMipLevel;
        lCopyRegion.srcSubresource.layerCount     = aSourceRegion.mMipLevelCount;
        lCopyRegion.srcOffset                     = { 0, 0, 0 };

        lCopyRegion.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        lCopyRegion.dstSubresource.baseArrayLayer = aDestRegion.mBaseLayer;
        lCopyRegion.dstSubresource.mipLevel       = aDestRegion.mBaseMipLevel;
        lCopyRegion.dstSubresource.layerCount     = aDestRegion.mMipLevelCount;
        lCopyRegion.dstOffset                     = { 0, 0, 0 };

        lCopyRegion.extent.width  = aSourceRegion.mWidth;
        lCopyRegion.extent.height = aSourceRegion.mHeight;
        lCopyRegion.extent.depth  = aSourceRegion.mDepth;

        vkCmdCopyImage( mVkObject, aSource, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, aDestination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &lCopyRegion );
        ImageMemoryBarrier( aDestination, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, 1 );
    }

    void sVkCommandBufferObject::CopyImage( VkImage aSource, VkBuffer aDestination, vec_t<sImageRegion> aImageRegions,
                                            uint32_t aBufferOffset )
    {
        vec_t<VkBufferImageCopy> lBufferCopyRegions;

        for( auto const &lRegion : aImageRegions )
        {
            VkBufferImageCopy lBufferCopyRegion{};

            lBufferCopyRegion.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            lBufferCopyRegion.imageSubresource.mipLevel       = lRegion.mBaseMipLevel;
            lBufferCopyRegion.imageSubresource.baseArrayLayer = lRegion.mBaseLayer;
            lBufferCopyRegion.imageSubresource.layerCount     = lRegion.mLayerCount;
            lBufferCopyRegion.imageExtent.width               = lRegion.mWidth;
            lBufferCopyRegion.imageExtent.height              = lRegion.mHeight;
            lBufferCopyRegion.imageExtent.depth               = lRegion.mDepth;
            lBufferCopyRegion.bufferOffset                    = lRegion.mOffset + aBufferOffset;

            lBufferCopyRegions.push_back( lBufferCopyRegion );
        }

        vkCmdCopyImageToBuffer( mVkObject, aSource, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, aDestination,
                                static_cast<uint32_t>( lBufferCopyRegions.size() ), lBufferCopyRegions.data() );
    }

    void sVkCommandBufferObject::End() { VK_CHECK_RESULT( vkEndCommandBuffer( mVkObject ) ); }

    void sVkCommandBufferObject::SetSubmitFence( VkFence aFence ) { mSubmitFence = aFence; }

    void sVkCommandBufferObject::AddWaitSemaphore( VkSemaphore aSemaphore, VkPipelineStageFlags aWaitStages )
    {
        mSubmitWaitSemaphores.push_back( aSemaphore );
        mSubmitWaitSemaphoreStage.push_back( aWaitStages );
    }

    void sVkCommandBufferObject::AddSignalSemaphore( VkSemaphore aSemaphore ) { mSubmitSignalSemaphores.push_back( aSemaphore ); }

    void sVkCommandBufferObject::SubmitTo( VkQueue aQueue )
    {
        Cast<VkGraphicContext>( mGraphicContext )->WaitForFences( { mSubmitFence } );

        VkSubmitInfo lQueueSubmitInfo{};
        lQueueSubmitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        lQueueSubmitInfo.commandBufferCount = 1;
        lQueueSubmitInfo.pCommandBuffers    = &mVkObject;

        if( mSubmitWaitSemaphores.size() > 0 )
        {
            lQueueSubmitInfo.waitSemaphoreCount = mSubmitWaitSemaphores.size();
            lQueueSubmitInfo.pWaitSemaphores    = mSubmitWaitSemaphores.data();
            lQueueSubmitInfo.pWaitDstStageMask  = mSubmitWaitSemaphoreStage.data();
        }

        if( mSubmitSignalSemaphores.size() > 0 )
        {
            lQueueSubmitInfo.signalSemaphoreCount = mSubmitSignalSemaphores.size();
            lQueueSubmitInfo.pSignalSemaphores    = mSubmitSignalSemaphores.data();
        }

        Cast<VkGraphicContext>( mGraphicContext )->ResetFence( mSubmitFence );
        vkQueueSubmit( aQueue, 1, &lQueueSubmitInfo, mSubmitFence );
    }
} // namespace SE::Graphics