#pragma once

#include "BaseImage.h"

#include "Core/Math/Types.h"

#include "UI/UIContext.h"

namespace SE::Core
{
    class UIImageButton : public UIBaseImage
    {
      public:
        UIImageButton() = default;

        UIImageButton( path_t const &aImagePath, math::vec2 aSize );
        UIImageButton( ref_t<ISampler2D> aImage, math::vec2 aSize );
        UIImageButton( path_t const &aImagePath, math::vec2 aSize, std::function<void()> aOnClick );
        UIImageButton( ref_t<ISampler2D> aImage, math::vec2 aSize, std::function<void()> aOnClick );

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