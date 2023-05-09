#pragma once

#include "BaseImage.h"
#include "Component.h"

namespace SE::Core
{
    class UIDropdownButton : public UIComponent
    {
      public:
        UIDropdownButton() = default;

        UIDropdownButton( std::function<bool( bool )> aOnClick );

        void OnClick( std::function<bool( bool )> aOnClick );
        void OnChanged( std::function<void()> aOnChanged );

        bool IsActive();
        void SetActive( bool aValue );

        void SetActiveImage( UIBaseImage *aImage );
        void SetInactiveImage( UIBaseImage *aImage );

      private:
        std::function<bool( bool )> mOnClicked;
        std::function<void()>       mOnChanged;

        bool mActivated = false;

        UIBaseImage *mActiveImage   = nullptr;
        UIBaseImage *mInactiveImage = nullptr;

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
        static void *UIDropdownButton_Create();
        static void  UIDropdownButton_Destroy( void *aInstance );
        static void  UIDropdownButton_OnClicked( void *aInstance, void *aHandler );
        static void  UIDropdownButton_OnChanged( void *aInstance, void *aHandler );
        static bool  UIDropdownButton_IsActive( void *aInstance );
        static void  UIDropdownButton_SetActive( void *aInstance, bool aValue );
        static void  UIDropdownButton_SetActiveImage( void *aInstance, void *aImage );
        static void  UIDropdownButton_SetInactiveImage( void *aInstance, void *aImage );
    };
} // namespace SE::Core