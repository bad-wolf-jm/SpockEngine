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

        void Add( UIComponent *aChild, string_t const &aKey );
        void SetCurrent( string_t const &aKey );

        ImVec2 RequiredSize();

      protected:
        string_t mCurrent;

        std::unordered_map<string_t, UIComponent *> mChildren;

      protected:
        void PushStyles();
        void PopStyles();

        void DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core