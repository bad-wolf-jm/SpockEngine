#pragma once

#include <memory>
#include <vulkan/vulkan.h>

#include "Core/Memory.h"

#include "VkTexture2D.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    class VkRenderTarget
    {
      public:
        VkFramebuffer mVkFramebuffer = VK_NULL_HANDLE;

      public:
        VkRenderTarget()                   = default;
        VkRenderTarget( VkRenderTarget & ) = default;
        VkRenderTarget( GraphicContext &aGraphicContext, uint32_t aWidth, uint32_t aHeight, uint32_t aLayers, VkRenderPass aRenderPass,
                        std::vector<Ref<VkTexture2D>> aAttachments );

        ~VkRenderTarget();

      private:
        GraphicContext mGraphicContext{};

        std::vector<VkImageView> mVkImageViews = {};
    };

} // namespace SE::Graphics
