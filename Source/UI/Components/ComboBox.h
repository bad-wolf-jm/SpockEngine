#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIComboBox : public UIComponent
    {
      public:
        UIComboBox() = default;

        UIComboBox( std::vector<std::string> const &aItems );

        void SetItemList(std::vector<std::string> aItems);

      void OnChange( std::function<void(int aIndex)> aOnChange );
      
      private:
        std::function<void(int aIndex)> mOnChange;

      protected:
        uint32_t                 mCurrentItem = 0;
        std::vector<std::string> mItems       = {};
        bool                     mChanged     = false;

        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );


    };

} // namespace SE::Core