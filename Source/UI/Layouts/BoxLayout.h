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

    class UIBoxLayout : public UIComponent
    {
      public:
        UIBoxLayout() = default;
        UIBoxLayout( eBoxLayoutOrientation aOrientation );

        UIBoxLayout( UIBoxLayout const & ) = default;

        ~UIBoxLayout() = default;

        void SetItemSpacing( float aItemSpacing );

        void Add( UIComponent *aChild, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                  eVerticalAlignment const &aVAlignment );
        void Add( UIComponent *aChild, bool aExpand, bool aFill );
        void Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill );
        void Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
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