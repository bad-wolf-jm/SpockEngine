#pragma once

#include "UI/Components/Component.h"

namespace SE::Core
{

    enum eBoxLayoutOrientation : int32_t
    {
        HORIZONTAL,
        VERTICAL
    };

    struct BoxLayoutItem
    {
        UIComponent *mItem = nullptr;

        float mFixedSize   = 0.0f;
        bool  mExpand      = true;
        bool  mFill        = true;
        bool  mIsSeparator = true;

        eHorizontalAlignment mHalign = eHorizontalAlignment::CENTER;
        eVerticalAlignment   mValign = eVerticalAlignment::CENTER;

        ImVec4 mSeparatorColor;

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

        void SetOrientation( eBoxLayoutOrientation aValue );
        void SetItemSpacing( float aItemSpacing );
        void SetSimple( bool aSimple )
        {
            mSimple = aSimple;
        }
        bool IsSimple( bool aSimple )
        {
            return mSimple;
        }

        void Add( UIComponent *aChild, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                  eVerticalAlignment const &aVAlignment );
        void Add( UIComponent *aChild, bool aExpand, bool aFill );
        void Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill );
        void Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                  eVerticalAlignment const &aVAlignment );

        void AddSeparator();

        void Clear();

        ImVec2 RequiredSize();

      protected:
        vector_t<BoxLayoutItem> mChildren;
        eBoxLayoutOrientation   mOrientation = eBoxLayoutOrientation::HORIZONTAL;
        float                   mItemSpacing = 0.0f;
        bool                    mSimple      = false;

      protected:
        void PushStyles();
        void PopStyles();

        void DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core