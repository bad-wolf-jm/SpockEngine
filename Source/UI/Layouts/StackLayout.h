#pragma once

#include "UI/Components/Component.h"

namespace SE::Core
{
    struct StackLayoutItem
    {
        UIComponent *mItem = nullptr;

        ImVec2 mSize{};
        ImVec2 mPosition{};

        bool mExpand = true;
        bool mFill   = true;

        eHorizontalAlignment mHalign = eHorizontalAlignment::CENTER;
        eVerticalAlignment   mValign = eVerticalAlignment::CENTER;

        StackLayoutItem()                          = default;
        StackLayoutItem( StackLayoutItem const & ) = default;

        ~StackLayoutItem() = default;
    };

    class UIStackLayout : public UIComponent
    {
      public:
        UIStackLayout() = default;

        UIStackLayout( UIStackLayout const & ) = default;

        ~UIStackLayout() = default;

        void Add( UIComponent *aChild, std::string const &aKey );
        void SetCurrent( std::string const &aKey );

        ImVec2 RequiredSize();
        
      protected:
        std::string mCurrent;

        std::unordered_map<std::string, UIComponent *> mChildren;

      protected:
        void PushStyles();
        void PopStyles();

        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

    //   public:
    //     static void *UIStackLayout_Create();
    //     static void  UIStackLayout_Destroy( void *aInstance );

    //     static void UIStackLayout_Add( void *aInstance, void *aChild, void *aKey );
    //     static void UIStackLayout_SetCurrent( void *aInstance, void *aKey );
    };
} // namespace SE::Core