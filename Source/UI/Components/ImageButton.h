#pragma once

#include "BaseImage.h"

#include "Core/Math/Types.h"

#include "Core/GraphicContext/UI/UIContext.h"

namespace SE::Core
{
    class UIImageButton : public UIBaseImage
    {
      public:
        UIImageButton() = default;

        UIImageButton( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize );
        UIImageButton( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize, std::function<void()> aOnClick );

        void OnClick( std::function<void()> aOnClick );

      private:
        std::function<void()> mOnClick;

      private:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core