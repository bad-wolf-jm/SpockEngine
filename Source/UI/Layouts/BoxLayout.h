#pragma once

#include "UI/Components/Component.h"

namespace SE::Core
{

    enum eBoxLayoutOrientation : uint8_t
    {
        HORIZONTAL,
        VERTICAL
    };
    struct BoxLayoutItem
    {
        UIComponent *mItem = nullptr;

        float mFixedSize = 0.0f;
        bool  mExpand    = true;
        bool  mFill      = true;

        eHorizontalAlignment mHalign = eHorizontalAlignment::CENTER;
        eVerticalAlignment   mValign = eVerticalAlignment::CENTER;

        BoxLayoutItem()                        = default;
        BoxLayoutItem( BoxLayoutItem const & ) = default;

        ~BoxLayoutItem() = default;
    };

    class BoxLayout : public UIComponent
    {
      public:
        BoxLayout() = default;
        BoxLayout( eBoxLayoutOrientation aOrientation );

        BoxLayout( BoxLayout const & ) = default;

        ~BoxLayout() = default;

        BoxLayout &SetItemSpacing( float aItemSpacing );

        BoxLayout &Add( UIComponent *aChild, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                        eVerticalAlignment const &aVAlignment );
        BoxLayout &Add( UIComponent *aChild, bool aExpand, bool aFill );
        BoxLayout &Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill );
        BoxLayout &Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                        eVerticalAlignment const &aVAlignment );

      protected:
        std::vector<BoxLayoutItem> mChildren;
        eBoxLayoutOrientation      mOrientation;
        float                      mItemSpacing = 0.0f;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core