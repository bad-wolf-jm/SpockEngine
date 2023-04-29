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
        IRenderPass( Ref<IGraphicContext> aContext, uint32_t aSampleCount );

        IRenderPass( IRenderPass & ) = default;

        ~IRenderPass() = default;

      protected:
        Ref<IGraphicContext> mGraphicContext = nullptr;
        uint32_t mSampleCount = 1;
    };
} // namespace SE::Graphics