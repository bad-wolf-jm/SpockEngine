#pragma once

#include "Component.h"
#include "Label.h"

namespace SE::Core
{
    class UICheckBox : public UIComponent
    {
      public:
        UICheckBox() = default;
        UICheckBox( std::function<void()> aOnClick );

        void OnClick( std::function<void()> aOnClick );

        bool IsChecked() { return mIsChecked; }
        void SetIsChecked(bool aValue) { mIsChecked = aValue; }

      private:
        std::function<void()> mOnClick;
        bool                  mIsChecked = false;

      private:
        void PushStyles();
        void PopStyles();

        void PushStyles( bool aEnabled );
        void PopStyles( bool aEnabled );

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UICheckBox_Create();
        static void  UICheckBox_Destroy( void *aInstance );
        static void  UICheckBox_OnClick( void *aInstance, math::vec4 *aTextColor );
        static bool  UICheckBox_IsChecked( void *aInstance );
        static void  UICheckBox_SetIsChecked( void *aInstance, bool aValue );
    };
} // namespace SE::Core