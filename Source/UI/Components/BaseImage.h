#pragma once

#include "Component.h"

#include "Core/Math/Types.h"

#include "UI/UIContext.h"

namespace SE::Core
{
    class UIBaseImage : public UIComponent
    {
      public:
        UIBaseImage() = default;

        UIBaseImage( path_t const &aImagePath, math::vec2 aSize );
        UIBaseImage( ref_t<ISampler2D> aImage, math::vec2 aSize );

        void SetImage( path_t const &aImagePath );

        ImVec2 Size();
        void   SetSize( float aWidth, float aHeight );
        void   SetSize( math::vec2 aSize );

        ImVec2 TopLeft();
        void   SetTopLeft( math::vec2 aTopLeft );

        ImVec2 BottomRight();
        void   SetBottomRight( math::vec2 aBottomRight );

        ImVec4 TintColor();
        void   SetTintColor( math::vec4 aColor );

        ImTextureID TextureID();

      protected:
        path_t mImagePath;

        ref_t<ISampler2D> mImage;
        ImageHandle       mHandle;

        ImVec2 mSize{};
        ImVec2 mTopLeft{};
        ImVec2 mBottomRight{};
        ImVec4 mTintColor{ 1.0f, 1.0f, 1.0f, 1.0f };

      private:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core