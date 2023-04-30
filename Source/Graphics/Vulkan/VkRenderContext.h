#pragma once

#include "Graphics/Interface/IGraphicContext.h"
#include "Graphics/Interface/IRenderContext.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"
#include "VkBaseRenderContext.h"
#include "VkRenderTarget.h"

namespace SE::Graphics
{
    class VkRenderContext : public VkBaseRenderContext
    {
      public:
        VkRenderContext() = default;
        VkRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<IRenderTarget> aRenderTarget );

        ~VkRenderContext() = default;

        bool BeginRender();
    };

} // namespace SE::Graphics