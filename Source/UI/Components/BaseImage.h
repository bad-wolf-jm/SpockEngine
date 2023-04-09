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

        UIBaseImage( fs::path const &aImagePath, math::vec2 aSize );
        UIBaseImage( Ref<VkSampler2D> aImage, math::vec2 aSize );

        void   SetImage( fs::path const &aImagePath );
        void   SetSize( float aWidth, float aHeight );
        void   SetRect( math::vec2 aTopLeft, math::vec2 aBottomRight );
        ImVec2 TopLeft();
        ImVec2 BottomRight();

        void   SetTintColor( math::vec4 aColor );
        ImVec4 TintColor();

        ImTextureID TextureID();
        ImVec2      Size();

      protected:
        fs::path mImagePath;

        Ref<VkSampler2D> mImage;
        ImageHandle      mHandle;

        ImVec2 mSize{};
        ImVec2 mTopLeft{};
        ImVec2 mBottomRight{};
        ImVec4 mTintColor{ 1.0f, 1.0f, 1.0f, 1.0f };

      private:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UIBaseImage_Create();
        static void *UIBaseImage_CreateWithPath( void* aText, math::vec2 aSize );
        static void  UIBaseImage_Destroy( void *aInstance );
        static void  UIBaseImage_SetImage( void *aInstance, void* aPath );
        static void  UIBaseImage_SetSize( void *aInstance, float aWidth, float aHeight );
        static void  UIBaseImage_SetRect( void *aInstance, math::vec2 aTopLeft, math::vec2 aBottomRight );
        static void  UIBaseImage_SetTintColor( void *aInstance, math::vec4 aColor );
    };
} // namespace SE::Core