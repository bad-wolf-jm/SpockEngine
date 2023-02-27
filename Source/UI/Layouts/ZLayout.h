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

        float mFixedSize = 0.0f;
        bool  mExpand    = true;
        bool  mFill      = true;

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
        UIZLayout( eZLayoutOrientation aOrientation );

        UIZLayout( UIZLayout const & ) = default;

        ~UIZLayout() = default;

        void SetItemSpacing( float aItemSpacing );

        void Add( UIComponent *aChild, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                  eVerticalAlignment const &aVAlignment );
        void Add( UIComponent *aChild, bool aExpand, bool aFill );
        void Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill );
        void Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                  eVerticalAlignment const &aVAlignment );

      protected:
        std::vector<ZLayoutItem> mChildren;
        eZLayoutOrientation    mOrientation;
        float                    mItemSpacing = 0.0f;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core