#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIComboBox : public UIComponent
    {
      public:
        UIComboBox() = default;

        UIComboBox( std::vector<std::string> const &aItems );

        void SetItemList( std::vector<std::string> aItems );

        void OnChange( std::function<void( int aIndex )> aOnChange );
        int  Current() { return mCurrentItem; }
        void SetCurrent(int aCurrent) { mCurrentItem = aCurrent; }

      private:
        std::function<void( int aIndex )> mOnChange;

      protected:
        uint32_t                 mCurrentItem = 0;
        std::vector<std::string> mItems       = {};
        bool                     mChanged     = false;

        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        void *mOnChangeDelegate       = nullptr;
        int   mOnChangeDelegateHandle = -1;

    //   public:
    //     static void *UIComboBox_Create();
    //     static void *UIComboBox_CreateWithItems( void *aItems );
    //     static void  UIComboBox_Destroy( void *aInstance );
    //     static int   UIComboBox_GetCurrent( void *aInstance );
    //     static void  UIComboBox_SetCurrent( void *aInstance, int aValue );
    //     static void  UIComboBox_SetItemList( void *aInstance, void *aItems );
    //     static void  UIComboBox_OnChanged( void *aInstance, void *aDelegate );
    };

} // namespace SE::Core