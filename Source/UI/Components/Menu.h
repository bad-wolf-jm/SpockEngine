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
        ImVec4      mTextColor;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      private:
        void *mOnTriggerDelegate       = nullptr;
        int   mOnTriggerDelegateHandle = -1;

      public:
        static void *UIMenuItem_Create();
        static void *UIMenuItem_CreateWithText( void *aText );
        static void *UIMenuItem_CreateWithTextAndShortcut( void *aText, void *aShortcut );
        static void  UIMenuItem_Destroy( void *aInstance );
        static void  UIMenuItem_SetText( void *aInstance, void *aText );
        static void  UIMenuItem_SetShortcut( void *aInstance, void *aShortcut );
        static void  UIMenuItem_SetTextColor( void *aInstance, math::vec4 *aTextColor );
        static void  UIMenuItem_OnTrigger( void *aInstance, void *aDelegate );
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

      public:
        static void *UIMenuSeparator_Create();
        static void  UIMenuSeparator_Destroy( void *aInstance );
    };

    class UIMenu : public UIMenuItem
    {
      public:
        UIMenu()                 = default;
        UIMenu( UIMenu const & ) = default;
        UIMenu( string_t const &aText );

        ref_t<UIMenuItem> AddAction( string_t const &aText, string_t const &aShortcut );
        ref_t<UIMenu>     AddMenu( string_t const &aText );
        ref_t<UIMenuItem> AddSeparator();

        void Update();

      private:
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

      public:
        static void *UIMenu_Create();
        static void *UIMenu_CreateWithText( void *aText );
        static void  UIMenu_Destroy( void *aInstance );
        static void *UIMenu_AddAction( void *aInstance, void *aText, void *aShortcut );
        static void *UIMenu_AddMenu( void *aInstance, void *aText );
        static void *UIMenu_AddSeparator( void *aInstance );
        static void  UIMenu_Update( void *aInstance );
    };
} // namespace SE::Core