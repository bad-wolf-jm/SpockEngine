#pragma once

#include "Component.h"

#include "Core/Math/Types.h"

#include "Core/GraphicContext/UI/UIContext.h"

namespace SE::Core
{
    class UIImage : public UIComponent
    {
      public:
        UIImage() = default;

        UIImage( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize );

        UIImage &SetImage( fs::path const &aImagePath );
        UIImage &SetSize( float aWidth, float aHeight );
        UIImage &SetBackgroundColor( math::vec4 aColor );
        UIImage &SetTintColor( math::vec4 aColor );

      private:
        Ref<UIContext> mUIContext;
        fs::path       mImagePath;

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