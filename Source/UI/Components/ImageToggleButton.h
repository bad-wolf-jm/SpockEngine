#pragma once

#include "BaseImage.h"
#include "Component.h"

namespace SE::Core
{
    class UIImageToggleButton : public UIComponent
    {
      public:
        UIImageToggleButton() = default;

        UIImageToggleButton( std::function<bool( bool )> aOnClick );

        void OnChange( std::function<bool( bool )> aOnClick );

        bool IsActive();
        void SetActive( bool aValue );

        void SetActiveImage( UIBaseImage *aImage );
        void SetInactiveImage( UIBaseImage *aImage );

      private:
        std::function<bool( bool )> mOnChange;

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

      public:
        static void *UIImageToggleButton_Create();
        static void  UIImageToggleButton_Destroy( void *aInstance );
        static void  UIImageToggleButton_OnChanged( void *aInstance, void *aHandler );
        static bool  UIImageToggleButton_IsActive( void *aInstance );
        static void  UIImageToggleButton_SetActive( void *aInstance, bool aValue );
        static void  UIImageToggleButton_SetActiveImage( void *aInstance, void *aImage );
        static void  UIImageToggleButton_SetInactiveImage( void *aInstance, void *aImage );
    };
} // namespace SE::Core