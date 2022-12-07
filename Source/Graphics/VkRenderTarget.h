#pragma once

#include <memory>
#include <vulkan/vulkan.h>

#include "Core/Memory.h"
#include "VkContext.h"
#include "VkTexture2D.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    class VkRenderTarget
    {
        VkRenderTarget()                   = default;
        VkRenderTarget( VkRenderTarget & ) = default;
        sVkFramebufferObject( Ref<VkContext> aContext, uint32_t aWidth, uint32_t aHeight, uint32_t aLayers, VkRenderPass aRenderPass,
                              std::vector<Ref<VkTexture2D>> aAttachments );

        ~VkRenderTarget();

      private:
        VkFramebuffer  mVkObject = VK_NULL_HANDLE;
        Ref<VkContext> mContext  = nullptr;
    };

} // namespace SE::Graphics
