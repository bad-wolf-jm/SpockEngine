#pragma once

#include "Component.h"

#include "Core/Math/Types.h"

#include "Core/GraphicContext/UI/UIContext.h"

namespace SE::Core
{
    class UIBaseImage : public UIComponent
    {
      public:
        UIBaseImage() = default;

        UIBaseImage( fs::path const &aImagePath, math::vec2 aSize );
        UIBaseImage( Ref<VkSampler2D> aImage, math::vec2 aSize );

        void SetImage( fs::path const &aImagePath );
        void SetSize( float aWidth, float aHeight );
        void SetRect( math::vec2 aTopLeft, math::vec2 aBottomRight );
        void SetBackgroundColor( math::vec4 aColor );
        void SetTintColor( math::vec4 aColor );

      protected:
        // Ref<UIContext> mUIContext;
        fs::path       mImagePath;

        Ref<VkSampler2D> mImage;
        ImageHandle      mHandle;

        ImVec2 mSize{};
        ImVec2 mTopLeft{};
        ImVec2 mBottomRight{};
        ImVec4 mBackgroundColor{ 0.0f, 0.0f, 0.0f, 0.0f };
        ImVec4 mTintColor{ 1.0f, 1.0f, 1.0f, 1.0f };

      private:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core