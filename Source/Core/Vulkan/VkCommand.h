#pragma once

#include "Core/Memory.h"
#include "Core/Platform/ViewportClient.h"
#include <memory>

#include "VkContext.h"
#include "VkImage.h"

#include "VkPipeline.h"
#include "VkRenderPass.h"

namespace LTSE::Graphics::Internal
{
    using namespace LTSE::Core;

    struct sVkCommandBufferObject
    {
        VkCommandBuffer mVkObject = VK_NULL_HANDLE;

        sVkCommandBufferObject()                           = default;
        sVkCommandBufferObject( sVkCommandBufferObject & ) = default;
        sVkCommandBufferObject( Ref<VkContext> aContext );
        sVkCommandBufferObject( Ref<VkContext> aContext, VkCommandBuffer aCommandBuffer );

        ~sVkCommandBufferObject();

        void Begin();
        void Begin( VkCommandBufferUsageFlags aUsage );

        void BeginRenderPass( Ref<sVkRenderPassObject> aRenderPass, Ref<sVkFramebufferObject> aFrameBuffer,
            math::uvec2 aExtent, std::vector<VkClearValue> aClearValues );
        void EndRenderPass();

        void SetViewport( math::ivec2 aOffset, math::uvec2 aSize );
        void SetScissor( math::ivec2 aOffset, math::uvec2 aSize );
        void Draw( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset, uint32_t aInstanceCount,
            uint32_t aFirstInstance );
        void DrawIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset,
            uint32_t aInstanceCount, uint32_t aFirstInstance );
        void Bind( Ref<sVkPipelineObject> aGraphicPipeline, VkPipelineBindPoint aBindPoint );
        void Bind( VkBuffer aVertexBuffer, uint32_t aBindPoint );
        void Bind( VkBuffer aVertexBuffer, VkBuffer aIndexBuffer, uint32_t aBindPoint );
        void Bind( Ref<sVkDescriptorSetObject> aDescriptorSet, VkPipelineBindPoint aBindPoint,
            Ref<sVkPipelineLayoutObject> aPipelineLayout, uint32_t aSetIndex, int32_t aDynamicOffset );

        void ImageMemoryBarrier( Ref<sVkImageObject> aImage, VkImageLayout aOldLayout, VkImageLayout aNewLayout,
            uint32_t aMipCount, uint32_t aLayerCount );

        void CopyBuffer( VkBuffer aSource, VkBuffer aDest );
        void CopyBuffer( VkBuffer aSource, uint32_t aSourceOffset, uint32_t aSize, VkBuffer aDest, uint32_t aDestOffset );
        void CopyBuffer( VkBuffer aSource, Ref<sVkImageObject> aDestination, sImageRegion const &aBufferRegion,
            sImageRegion const &aImageRegion );
        void CopyBuffer( VkBuffer aSource, Ref<sVkImageObject> aDestination, std::vector<sImageRegion> aBufferRegions,
            sImageRegion const &aImageRegion );

        void CopyImage( Ref<sVkImageObject> aSource, sImageRegion const &aSourceRegion, Ref<sVkImageObject> aDestination,
            sImageRegion const &aDestRegion );

        void CopyImage( Ref<sVkImageObject> aSource, sImageRegion const &aSourceRegion, VkBuffer aDestination,
            sImageRegion const &aDestRegion );

        template <typename T>
        void PushConstants( VkShaderStageFlags aShaderStages, uint32_t aOffset, const T &aValue,
            Ref<sVkPipelineLayoutObject> aPipelineLayout )
        {
            vkCmdPushConstants(
                mVkObject, aPipelineLayout->mVkObject, aShaderStages, aOffset, sizeof( T ), (void *)&aValue );
        }

        void End();

        void SetSubmitFence( VkFence aFence );
        void AddWaitSemaphore( VkSemaphore aSemaphore, VkPipelineStageFlags aWaitStages );
        void AddSignalSemaphore( VkSemaphore aSemaphore );

        void SubmitTo( VkQueue aQueue );

      private:
        Ref<VkContext>                    mContext                  = nullptr;
        VkFence                           mSubmitFence              = nullptr;
        std::vector<VkSemaphore>          mSubmitWaitSemaphores     = {};
        std::vector<VkPipelineStageFlags> mSubmitWaitSemaphoreStage = {};
        std::vector<VkSemaphore>          mSubmitSignalSemaphores   = {};
    };
} // namespace LTSE::Graphics::Internal
