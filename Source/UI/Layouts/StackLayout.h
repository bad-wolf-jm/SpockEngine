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

    class StackLayout : public UIComponent
    {
      public:
        StackLayout() = default;

        StackLayout( StackLayout const & ) = default;

        ~StackLayout() = default;

        void Add( UIComponent *aChild, std::string const &aKey );
        void SetCurrent( std::string const &aKey );

      protected:
        std::string mCurrent;

        std::unordered_map<std::string, UIComponent *> mChildren;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *StackLayout_Create();
        static void  StackLayout_Destroy( void *aInstance );

        static void StackLayout_Add( void *aInstance, void *aChild, void *aKey );
        static void StackLayout_SetCurrent( void *aInstance, void *aKey );
    };
} // namespace SE::Core