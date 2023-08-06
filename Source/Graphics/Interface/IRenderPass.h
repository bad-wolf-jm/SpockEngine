#pragma once

#include "Core/Memory.h"

#include "IGraphicContext.h"
#include "ITexture.h"
#include "ITextureCubeMap.h"

namespace SE::Graphics
{
    class IRenderPass
    {
      public:
        IRenderPass() = default;
        IRenderPass( ref_t<IGraphicContext> aContext, uint32_t aSampleCount );

        IRenderPass( IRenderPass & ) = default;

        ~IRenderPass() = default;

      protected:
        ref_t<IGraphicContext> mGraphicContext = nullptr;
        uint32_t               mSampleCount    = 1;
    };
} // namespace SE::Graphics