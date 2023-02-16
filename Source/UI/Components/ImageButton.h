#pragma once

#include "Component.h"

#include "Core/Math/Types.h"

#include "Core/GraphicContext/UI/UIContext.h"

namespace SE::Core
{
    class UIImageButton : public UIComponent
    {
      public:
        UIImageButton() = default;

        UIImageButton( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize );
        UIImageButton( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize, std::function<void()> aOnClick );

        UIImageButton &SetImage( fs::path const &aImagePath );
        UIImageButton &SetSize( float aWidth, float aHeignt );
        UIImageButton &SetBackgroundColor( math::vec4 aColor );
        UIImageButton &SetTintColor( math::vec4 aColor );

      private:
        Ref<UIContext>        mUIContext;
        fs::path              mImagePath;
        std::function<void()> mOnClick;

        Ref<VkSampler2D> mImage;
        ImageHandle      mHandle;

        ImVec2 mSize{};
        ImVec4 mBackgroundColor{ 0.0f, 0.0f, 0.0f, 0.0f };
        ImVec4 mTintColor{ 1.0f, 1.0f, 1.0f, 1.0f };

      private:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core