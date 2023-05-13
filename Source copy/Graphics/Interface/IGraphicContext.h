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
        IGraphicContext( Ref<IWindow> aWindow, uint32_t aSampleCount = 1 )
            : mWindow{ aWindow }
            , mSampleCount{ aSampleCount }
        {
        }

        Ref<IWindow> GetWindow() { return mWindow; };

        virtual eColorFormat GetDepthFormat() = 0;

      protected:
        Ref<IWindow> mWindow = nullptr;

        uint32_t mSampleCount = 1;
    };
} // namespace SE::Graphics
