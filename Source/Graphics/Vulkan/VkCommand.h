#pragma once

#include "Core/Memory.h"

#include <memory>

#include "Graphics/Interface/ICommandBuffer.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
// #include "VkImage.h"

#include "Graphics/Vulkan/VkPipeline.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    struct sImageRegion
    {
        uint32_t mBaseMipLevel  = 0;
        uint32_t mMipLevelCount = 0;
        uint32_t mBaseLayer     = 0;
        uint32_t mLayerCount    = 0;
        uint32_t mOffset        = 0;
        uint32_t mWidth         = 0;
        uint32_t mHeight        = 0;
        uint32_t mDepth         = 0;
    };

    struct sVkCommandBufferObject : public ICommandBuffer
    {
        VkCommandBuffer mVkObject = VK_NULL_HANDLE;

        sVkCommandBufferObject()                           = default;
        sVkCommandBufferObject( sVkCommandBufferObject & ) = default;
        sVkCommandBufferObject( ref_t<VkGraphicContext> aContext );
        sVkCommandBufferObject( ref_t<VkGraphicContext> aContext, VkCommandBuffer aCommandBuffer );

        ~sVkCommandBufferObject();

        void Begin();
        void Begin( VkCommandBufferUsageFlags aUsage );

        void BeginRenderPass( ref_t<IRenderPass> aRenderPass, VkFramebuffer aFrameBuffer, math::uvec2 aExtent,
                              vector_t<VkClearValue> aClearValues );
        void EndRenderPass();

        void SetViewport( math::ivec2 aOffset, math::uvec2 aSize );
        void SetScissor( math::ivec2 aOffset, math::uvec2 aSize );
        void Draw( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset, uint32_t aInstanceCount,
                   uint32_t aFirstInstance );
        void DrawIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset, uint32_t aInstanceCount,
                          uint32_t aFirstInstance );
        void Bind( ref_t<sVkPipelineObject> aGraphicPipeline, VkPipelineBindPoint aBindPoint );
        void Bind( VkBuffer aVertexBuffer, uint32_t aBindPoint );
        void Bind( VkBuffer aVertexBuffer, VkBuffer aIndexBuffer, uint32_t aBindPoint );
        void Bind( ref_t<sVkDescriptorSetObject> aDescriptorSet, VkPipelineBindPoint aBindPoint,
                   ref_t<sVkPipelineLayoutObject> aPipelineLayout, uint32_t aSetIndex, int32_t aDynamicOffset );
        void Bind( void *aDescriptorSet, VkPipelineBindPoint aBindPoint, ref_t<sVkPipelineLayoutObject> aPipelineLayout,
                   uint32_t aSetIndex, int32_t aDynamicOffset );

        void ImageMemoryBarrier( VkImage aImage, VkImageLayout aOldLayout, VkImageLayout aNewLayout, uint32_t aMipCount,
                                 uint32_t aLayerCount );

        void CopyBuffer( VkBuffer aSource, VkBuffer aDest );
        void CopyBuffer( VkBuffer aSource, uint32_t aSourceOffset, uint32_t aSize, VkBuffer aDest, uint32_t aDestOffset );

        void CopyBuffer( VkBuffer aSource, VkImage aDestination, sImageRegion const &aBufferRegion, sImageRegion const &aImageRegion );
        void CopyBuffer( VkBuffer aSource, VkImage aDestination, vector_t<sImageRegion> aBufferRegions,
                         sImageRegion const &aImageRegion );

        void CopyImage( VkImage aSource, sImageRegion const &aSourceRegion, VkImage aDestination, sImageRegion const &aDestRegion );
        void CopyImage( VkImage aSource, VkBuffer aDestination, vector_t<sImageRegion> aImageRegions, uint32_t aBufferOffset );

        template <typename T>
        void PushConstants( VkShaderStageFlags aShaderStages, uint32_t aOffset, const T &aValue,
                            ref_t<sVkPipelineLayoutObject> aPipelineLayout )
        {
            vkCmdPushConstants( mVkObject, aPipelineLayout->mVkObject, aShaderStages, aOffset, sizeof( T ), (void *)&aValue );
        }

        void PushConstants( VkShaderStageFlags aShaderStages, uint32_t aOffset, void *aValue, uint32_t aSize,
                            ref_t<sVkPipelineLayoutObject> aPipelineLayout )
        {
            vkCmdPushConstants( mVkObject, aPipelineLayout->mVkObject, aShaderStages, aOffset, aSize, aValue );
        }

        void End();

        void SetSubmitFence( VkFence aFence );
        void AddWaitSemaphore( VkSemaphore aSemaphore, VkPipelineStageFlags aWaitStages );
        void AddSignalSemaphore( VkSemaphore aSemaphore );

        void SubmitTo( VkQueue aQueue );

      private:
        VkFence                        mSubmitFence              = nullptr;
        vector_t<VkSemaphore>          mSubmitWaitSemaphores     = {};
        vector_t<VkPipelineStageFlags> mSubmitWaitSemaphoreStage = {};
        vector_t<VkSemaphore>          mSubmitSignalSemaphores   = {};
    };
} // namespace SE::Graphics
