#pragma once

#include "BaseImage.h"

#include "Core/Math/Types.h"

#include "Core/GraphicContext/UI/UIContext.h"

namespace SE::Core
{
    class UIImage : public UIBaseImage
    {
      public:
        UIImage() = default;
        UIImage( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize );

      private:
        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core