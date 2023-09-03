#pragma once

#include "Graphics/Interface/IGraphicContext.h"
#include "Graphics/Interface/IRenderContext.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"
#include "VkBaseRenderContext.h"
#include "VkRenderTarget.h"

namespace SE::Graphics
{
    class VkSwapChainRenderContext : public VkBaseRenderContext
    {
      public:
        VkSwapChainRenderContext() = default;
        VkSwapChainRenderContext( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderTarget> aRenderTarget );

        ~VkSwapChainRenderContext() = default;

        bool BeginRender();
    };

} // namespace SE::Graphics