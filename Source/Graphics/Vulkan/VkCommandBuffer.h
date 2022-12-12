#pragma once

#include "Core/Memory.h"

#include <memory>

#include "VkGraphicContext.h"
// #include "VkImage.h"

#include "VkPipeline.h"

namespace SE::Graphics::Internal
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

    struct sVkCommandBuffer
    {
        VkCommandBuffer mVkObject = VK_NULL_HANDLE;

        sVkCommandBuffer()                     = default;
        sVkCommandBuffer( sVkCommandBuffer & ) = default;
        sVkCommandBuffer( Ref<VkGraphicContext> aContext );
        sVkCommandBuffer( Ref<VkGraphicContext> aContext, VkCommandBuffer aCommandBuffer );

        ~sVkCommandBuffer();

        void Begin();
        void Begin( VkCommandBufferUsageFlags aUsage );

        void BeginRenderPass( Ref<sVkAbstractRenderPassObject> aRenderPass, VkFramebuffer aFrameBuffer, math::uvec2 aExtent,
                              std::vector<VkClearValue> aClearValues );
        void EndRenderPass();

        void SetViewport( math::ivec2 aOffset, math::uvec2 aSize );
        void SetScissor( math::ivec2 aOffset, math::uvec2 aSize );
        void Draw( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset, uint32_t aInstanceCount,
                   uint32_t aFirstInstance );
        void DrawIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset, uint32_t aInstanceCount,
                          uint32_t aFirstInstance );
        void Bind( Ref<sVkPipelineObject> aGraphicPipeline, VkPipelineBindPoint aBindPoint );
        void Bind( VkBuffer aVertexBuffer, uint32_t aBindPoint );
        void Bind( VkBuffer aVertexBuffer, VkBuffer aIndexBuffer, uint32_t aBindPoint );
        void Bind( Ref<sVkDescriptorSetObject> aDescriptorSet, VkPipelineBindPoint aBindPoint,
                   Ref<sVkPipelineLayoutObject> aPipelineLayout, uint32_t aSetIndex, int32_t aDynamicOffset );

        void ImageMemoryBarrier( VkImage aImage, VkImageLayout aOldLayout, VkImageLayout aNewLayout, uint32_t aMipCount,
                                 uint32_t aLayerCount );

        void CopyBuffer( VkBuffer aSource, VkBuffer aDest );
        void CopyBuffer( VkBuffer aSource, uint32_t aSourceOffset, uint32_t aSize, VkBuffer aDest, uint32_t aDestOffset );

        void CopyBuffer( VkBuffer aSource, VkImage aDestination, sImageRegion const &aBufferRegion, sImageRegion const &aImageRegion );
        void CopyBuffer( VkBuffer aSource, VkImage aDestination, std::vector<sImageRegion> aBufferRegions,
                         sImageRegion const &aImageRegion );

        void CopyImage( VkImage aSource, sImageRegion const &aSourceRegion, VkImage aDestination, sImageRegion const &aDestRegion );
        void CopyImage( VkImage aSource, VkBuffer aDestination, std::vector<sImageRegion> aImageRegions, uint32_t aBufferOffset );

        template <typename T>
        void PushConstants( VkShaderStageFlags aShaderStages, uint32_t aOffset, const T &aValue,
                            Ref<sVkPipelineLayoutObject> aPipelineLayout )
        {
            vkCmdPushConstants( mVkObject, aPipelineLayout->mVkObject, aShaderStages, aOffset, sizeof( T ), (void *)&aValue );
        }

        void End();

        void SetSubmitFence( VkFence aFence );
        void AddWaitSemaphore( VkSemaphore aSemaphore, VkPipelineStageFlags aWaitStages );
        void AddSignalSemaphore( VkSemaphore aSemaphore );

        void SubmitTo( VkQueue aQueue );

      private:
        Ref<VkGraphicContext>             mGraphicContext           = nullptr;
        VkFence                           mSubmitFence              = nullptr;
        std::vector<VkSemaphore>          mSubmitWaitSemaphores     = {};
        std::vector<VkPipelineStageFlags> mSubmitWaitSemaphoreStage = {};
        std::vector<VkSemaphore>          mSubmitSignalSemaphores   = {};
    };
} // namespace SE::Graphics::Internal
