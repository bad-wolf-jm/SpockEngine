#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIComboBox : public UIComponent
    {
      public:
        UIComboBox() = default;

        UIComboBox( std::vector<string_t> const &aItems );

        void SetItemList( std::vector<string_t> aItems );

        void OnChange( std::function<void( int aIndex )> aOnChange );
        int  Current() { return mCurrentItem; }
        void SetCurrent( int aCurrent ) { mCurrentItem = aCurrent; }

      private:
        std::function<void( int aIndex )> mOnChange;

      protected:
        uint32_t              mCurrentItem = 0;
        std::vector<string_t> mItems       = {};
        bool                  mChanged     = false;

        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };

} // namespace SE::Core