#pragma once

#include "IWindow.h"

namespace SE::Graphics
{
    class IGraphicContext
    {
      public:
        IGraphicContext( Ref<IWindow> aWindow, uint32_t aSampleCount = 1 )
            : mWindow{ aWindow }
            , mSampleCount{ aSampleCount }
        {
        }

        Ref<IWindow> GetWindow() { return mWindow; };

      protected:
        Ref<IWindow> mWindow = nullptr;

        uint32_t mSampleCount = 1;
    }
} // namespace SE::Graphics
