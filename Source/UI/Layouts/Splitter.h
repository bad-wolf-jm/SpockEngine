#pragma once

#include "BoxLayout.h"
#include "UI/Components/Component.h"

namespace SE::Core
{

    struct SplitterItem
    {
        UIComponent *mItem = nullptr;

        bool mFill = true;

        eHorizontalAlignment mHalign = eHorizontalAlignment::CENTER;
        eVerticalAlignment   mValign = eVerticalAlignment::CENTER;

        SplitterItem()                       = default;
        SplitterItem( SplitterItem const & ) = default;

        ~SplitterItem() = default;
    };

    class UISplitter : public UIComponent
    {
      public:
        UISplitter() = default;
        UISplitter( eBoxLayoutOrientation aOrientation );

        UISplitter( UISplitter const & ) = default;

        ~UISplitter() = default;

        void SetItemSpacing( float aItemSpacing );

        void Add( UIComponent *aChild );
        void Add( UIComponent *aChild, eHorizontalAlignment const &aHAlignment, eVerticalAlignment const &aVAlignment );

      protected:
        std::vector<SplitterItem> mChildren;
        std::vector<float>        mItemSizes;
        eBoxLayoutOrientation     mOrientation;
        float                     mItemSpacing = 0.0f;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UISplitter_CreateWithOrientation( eBoxLayoutOrientation aOrientation );
        static void  UISplitter_Destroy( void *aInstance );

        static void UISplitter_AddFill( void *aInstance, void *aChild );
        static void UISplitter_AddAligned( void *aInstance, void *aChild, eHorizontalAlignment aHAlignment,
                                           eVerticalAlignment aVAlignment );
        static void UISplitter_SetItemSpacing( void *aInstance, float aItemSpacing );
    };
} // namespace SE::Core