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
        UIBaseImage( Ref<ISampler2D> aImage, math::vec2 aSize );

        void SetImage( fs::path const &aImagePath );

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
        fs::path mImagePath;

        Ref<ISampler2D> mImage;
        ImageHandle     mHandle;

        ImVec2 mSize{};
        ImVec2 mTopLeft{};
        ImVec2 mBottomRight{};
        ImVec4 mTintColor{ 1.0f, 1.0f, 1.0f, 1.0f };

      private:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

    //   public:
    //     static void *UIBaseImage_Create();
    //     static void *UIBaseImage_CreateWithPath( void *aText, math::vec2 aSize );
    //     static void  UIBaseImage_Destroy( void *aInstance );
    //     static void  UIBaseImage_SetImage( void *aInstance, void *aPath );

    //     static void       UIBaseImage_SetSize( void *aInstance, math::vec2 aSize );
    //     static math::vec2 UIBaseImage_GetSize( void *aInstance );
    //     static void       UIBaseImage_SetTopLeft( void *aInstance, math::vec2 aTopLeft );
    //     static math::vec2 UIBaseImage_GetTopLeft( void *aInstance );
    //     static void       UIBaseImage_SetBottomRight( void *aInstance, math::vec2 aBottomRight );
    //     static math::vec2 UIBaseImage_GetBottomRight( void *aInstance );
    //     static void       UIBaseImage_SetTintColor( void *aInstance, math::vec4 aColor );
    //     static math::vec4 UIBaseImage_GetTintColor( void *aInstance );
    };
} // namespace SE::Core