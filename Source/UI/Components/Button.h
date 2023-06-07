#pragma once

#include "Component.h"
#include "Label.h"

namespace SE::Core
{
    class UIButton : public UILabel
    {
      public:
        UIButton() = default;

        UIButton( std::string const &aText );
        UIButton( std::string const &aText, std::function<void()> aOnClick );

        void OnClick( std::function<void()> aOnClick );
        void SetText( std::string const &aText );

      private:
        std::function<void()> mOnClick;

      private:
        void PushStyles();
        void PopStyles();

        void PushStyles( bool aEnabled );
        void PopStyles( bool aEnabled );

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        void *mOnClickDelegate       = nullptr;
        int   mOnClickDelegateHandle = -1;

    //   public:
    //     static void *UIButton_Create();
    //     static void *UIButton_CreateWithText( void *aText );
    //     static void  UIButton_Destroy( void *aInstance );
    //     static void  UIButton_OnClick( void *aInstance, void *aDelegate );
    //     static void  UIButton_SetText( void *aInstance, void *aText );
    };
} // namespace SE::Core