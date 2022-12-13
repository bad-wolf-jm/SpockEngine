#pragma once

#include <memory>
#include <vulkan/vulkan.h>

#include "Core/Memory.h"

#include "VkGraphicContext.h"
#include "VkTexture2D.h"

namespace SE::Graphics
{
    using namespace SE::Core;
    using namespace SE::Graphics::Internal;

    class VkRenderTarget
    {
      public:
        VkFramebuffer mVkFramebuffer = VK_NULL_HANDLE;

      public:
        VkRenderTarget()                   = default;
        VkRenderTarget( VkRenderTarget & ) = default;
        VkRenderTarget( Ref<VkGraphicContext> aGraphicContext, uint32_t aWidth, uint32_t aHeight, uint32_t aLayers,
                        VkRenderPass aRenderPass, std::vector<Ref<VkTexture2D>> aAttachments );

        ~VkRenderTarget();

      private:
        Ref<VkGraphicContext> mGraphicContext{};

        std::vector<VkImageView> mVkImageViews = {};
    };

} // namespace SE::Graphics
