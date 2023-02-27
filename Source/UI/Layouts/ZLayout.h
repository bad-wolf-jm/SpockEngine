#pragma once

#include "UI/Components/Component.h"

namespace SE::Core
{

    enum eZLayoutOrientation : uint8_t
    {
        HORIZONTAL,
        VERTICAL
    };
    struct ZLayoutItem
    {
        UIComponent *mItem = nullptr;

        math::vec2 mFixedSize{};
        bool       mExpand    = true;
        bool       mFill      = true;

        eHorizontalAlignment mHalign = eHorizontalAlignment::CENTER;
        eVerticalAlignment   mValign = eVerticalAlignment::CENTER;

        ZLayoutItem()                      = default;
        ZLayoutItem( ZLayoutItem const & ) = default;

        ~ZLayoutItem() = default;
    };

    class UIZLayout : public UIComponent
    {
      public:
        UIZLayout() = default;

        UIZLayout( UIZLayout const & ) = default;

        ~UIZLayout() = default;

        void Add( UIComponent *aChild, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                  eVerticalAlignment const &aVAlignment );
        void Add( UIComponent *aChild, bool aExpand, bool aFill );
        void Add( UIComponent *aChild, math::vec2 aFixedSize, bool aExpand, bool aFill );
        void Add( UIComponent *aChild, math::vec2 aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                  eVerticalAlignment const &aVAlignment );

      protected:
        std::vector<ZLayoutItem> mChildren;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core