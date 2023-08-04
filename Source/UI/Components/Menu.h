#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIMenuItem : public UIComponent
    {
      public:
        UIMenuItem()                     = default;
        UIMenuItem( UIMenuItem const & ) = default;

        UIMenuItem( string_t const &aText );
        UIMenuItem( string_t const &aText, string_t const &aShortcut );

        void SetText( string_t const &aText );
        void SetShortcut( string_t const &aShortcut );
        void SetTextColor( math::vec4 aColor );

        void OnTrigger( std::function<void()> aOnTrigger );

      protected:
        std::function<void()> mOnTrigger;

      protected:
        string_t mText;
        string_t mShortcut;
        ImVec4   mTextColor;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };

    class UIMenuSeparator : public UIMenuItem
    {
      public:
        UIMenuSeparator()                          = default;
        UIMenuSeparator( UIMenuSeparator const & ) = default;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };

    class UIMenu : public UIMenuItem
    {
      public:
        UIMenu()                 = default;
        UIMenu( UIMenu const & ) = default;
        UIMenu( string_t const &aText );

        Ref<UIMenuItem> AddAction( string_t const &aText, string_t const &aShortcut );
        Ref<UIMenu>     AddMenu( string_t const &aText );
        Ref<UIMenuItem> AddSeparator();

        void Update();

      public:
        UIMenuItem *AddActionRaw( string_t const &aText, string_t const &aShortcut );
        UIMenu     *AddMenuRaw( string_t const &aText );
        UIMenuItem *AddSeparatorRaw();

      protected:
        vector_t<UIMenuItem *> mActions;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core