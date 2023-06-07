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

        void Add1( UIComponent *aChild );
        void Add2( UIComponent *aChild );
        
        ImVec2 RequiredSize();

      protected:
        UIComponent *mChild1;
        UIComponent *mChild2;

        bool  mSizeSet     = false;
        float mSize1       = 100.0f;
        float mSize2       = 100.0f;
        float mItemSpacing = 2.0f;

        ImVec2 mCurrentSize{};

        eBoxLayoutOrientation mOrientation;

      protected:
        void PushStyles();
        void PopStyles();

        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

    //   public:
    //     static void *UISplitter_Create();
    //     static void *UISplitter_CreateWithOrientation( eBoxLayoutOrientation aOrientation );
    //     static void  UISplitter_Destroy( void *aInstance );

    //     static void UISplitter_Add1( void *aInstance, void *aChild );
    //     static void UISplitter_Add2( void *aInstance, void *aChild );

    //     static void UISplitter_SetItemSpacing( void *aInstance, float aItemSpacing );
    };
} // namespace SE::Core