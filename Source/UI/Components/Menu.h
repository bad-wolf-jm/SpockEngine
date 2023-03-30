#pragma once

#include "Component.h"

namespace SE::Core
{

    class UIMenuItem : public UIComponent
    {
      public:
        UIMenuItem() = default;

        UIMenuItem( std::string const &aText );

        void SetText( std::string const &aText );
        void SetTextColor( math::vec4 aColor );

      protected:
        std::string mText;
        ImVec4      mTextColor;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UIMenuItem_Create();
        static void *UIMenuItem_CreateWithText( void *aText );
        static void *UIMenu_CreateWithTextAndShortcut( void *aText, void *aShortcut );
        static void  UIMenuItem_Destroy( void *aInstance );
        static void  UIMenuItem_SetText( void *aInstance, void *aText );
        static void  UIMenuItem_SetTextColor( void *aInstance, math::vec4 *aTextColor );
    };

    class UIMenu : public UIMenuItem
    {
      public:
        UIMenu() = default;

        UIMenu( std::string const &aText );

        void SetText( std::string const &aText );
        void SetTextColor( math::vec4 aColor );
        void SetShortcut( std::string const &aText );

        UIMenuItem *AddAction( std::string const &aText, std::string const &aShortcut );
        UIMenuItem *AddSeparator();
        UIMenuItem *AddMenu( std::string const &aText );

      protected:
        std::string mText;
        ImVec4      mTextColor;

        std::vector<UIMenuItem *> mItems;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UIMenu_Create();
        static void *UIMenu_CreateWithText( void *aText );
        static void *UIMenu_CreateWithTextAndShortcut( void *aText, void *aShortcut );
        static void  UIMenu_Destroy( void *aInstance );
        static void  UIMenu_SetText( void *aInstance, void *aText );
        static void  UIMenu_SetTextColor( void *aInstance, math::vec4 *aTextColor );
    }
} // namespace SE::Core