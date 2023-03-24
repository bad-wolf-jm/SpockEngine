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

      public:
        static void *UIBoxLayout_CreateWithOrientation( eBoxLayoutOrientation aOrientation );
        static void  UIBoxLayout_Destroy( void *aInstance );

        static void UIBoxLayout_AddAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill,
                                                    eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
        static void UIBoxLayout_AddNonAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill );
        static void UIBoxLayout_AddAlignedFixed( void *aInstance, void *aChild, float aFixedSize, bool aExpand, bool aFill,
                                                 eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
        static void UIBoxLayout_AddNonAlignedFixed( void *aInstance, void *aChild, float aFixedSize, bool aExpand, bool aFill );
        static void UIBoxLayout_SetItemSpacing( void *aInstance, float aItemSpacing );
    };
} // namespace SE::Core