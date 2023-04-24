#pragma once

#include "Core/Memory.h"

#include "Core/CUDA/Texture/TextureData.h"
#include "IWindow.h"

namespace SE::Graphics
{
    using namespace SE::Core;
    class IGraphicContext
    {
      public:
        IGraphicContext( uint32_t aSampleCount = 1 )
            : mSampleCount{ aSampleCount }
        {
        }

        virtual eColorFormat GetDepthFormat() = 0;

      protected:
        uint32_t mSampleCount = 1;
    };
} // namespace SE::Graphics
