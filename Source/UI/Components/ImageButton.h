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

        UIImageButton( fs::path const &aImagePath, math::vec2 aSize );
        UIImageButton( Ref<ISampler2D> aImage, math::vec2 aSize );
        UIImageButton( fs::path const &aImagePath, math::vec2 aSize, std::function<void()> aOnClick );
        UIImageButton( Ref<ISampler2D> aImage, math::vec2 aSize, std::function<void()> aOnClick );

        void OnClick( std::function<void()> aOnClick );

      private:
        std::function<void()> mOnClick;

      private:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        void *mOnClickDelegate       = nullptr;
        int   mOnClickDelegateHandle = -1;

    //   public:
    //     static void *UIImageButton_Create();
    //     static void *UIImageButton_CreateWithPath( void* aText, math::vec2 *aSize );
    //     static void  UIImageButton_Destroy( void *aInstance );
    //     static void  UIImageButton_OnClick( void *aInstance, void *aDelegate );
    };
} // namespace SE::Core