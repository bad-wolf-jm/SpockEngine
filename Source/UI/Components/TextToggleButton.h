#pragma once

#include "Component.h"
#include "Label.h"

namespace SE::Core
{
    class UITextToggleButton : public UILabel
    {
      public:
        UITextToggleButton() = default;

        UITextToggleButton( string_t const &aText );
        UITextToggleButton( string_t const &aText, std::function<bool( bool )> aOnClick );

        bool IsActive();
        void SetActive( bool aValue );

        void OnClick( std::function<bool( bool )> aOnClick );
        void OnChanged( std::function<void()> aOnChanged );

        void SetActiveColor( math::vec4 const &aColor );
        void SetInactiveColor( math::vec4 const &aColor );

      private:
        std::function<bool( bool )> mOnClicked;
        std::function<void()>       mOnChanged;

        bool   mActivated     = false;
        ImVec4 mActiveColor   = ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f };
        ImVec4 mInactiveColor = ImVec4{ 1.0f, 1.0f, 1.0f, .2f };

      private:
        void PushStyles();
        void PopStyles();

        void PushStyles( bool aEnabled );
        void PopStyles( bool aEnabled );

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      private:
        void *mOnChangeDelegate       = nullptr;
        int   mOnChangeDelegateHandle = -1;

        void *mOnClickDelegate       = nullptr;
        int   mOnClickDelegateHandle = -1;

      public:
        static void *UITextToggleButton_Create();
        static void *UITextToggleButton_CreateWithText( void *aText );
        static void  UITextToggleButton_Destroy( void *aInstance );
        static void  UITextToggleButton_OnClicked( void *aInstance, void *aHandler );
        static void  UITextToggleButton_OnChanged( void *aInstance, void *aHandler );
        static bool  UITextToggleButton_IsActive( void *aInstance );
        static void  UITextToggleButton_SetActive( void *aInstance, bool aValue );
        static void  UITextToggleButton_SetActiveColor( void *aInstance, math::vec4 *aColor );
        static void  UITextToggleButton_SetInactiveColor( void *aInstance, math::vec4 *aColor );
    };
} // namespace SE::Core