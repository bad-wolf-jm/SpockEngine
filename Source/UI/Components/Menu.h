#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIMenuItem : public UIComponent
    {
      public:
        UIMenuItem()                     = default;
        UIMenuItem( UIMenuItem const & ) = default;

        UIMenuItem( std::string const &aText );
        UIMenuItem( std::string const &aText, std::string const &aShortcut );

        void SetText( std::string const &aText );
        void SetShortcut( std::string const &aShortcut );
        void SetTextColor( math::vec4 aColor );

        void OnTrigger( std::function<void()> aOnTrigger );

      protected:
        std::function<void()> mOnTrigger;

      protected:
        std::string mText;
        std::string mShortcut;
        ImVec4      mTextColor;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        void *mOnTriggerDelegate       = nullptr;
        int   mOnTriggerDelegateHandle = -1;

    //   public:
    //     static void *UIMenuItem_Create();
    //     static void *UIMenuItem_CreateWithText( void *aText );
    //     static void *UIMenuItem_CreateWithTextAndShortcut( void *aText, void *aShortcut );
    //     static void  UIMenuItem_Destroy( void *aInstance );
    //     static void  UIMenuItem_SetText( void *aInstance, void *aText );
    //     static void  UIMenuItem_SetShortcut( void *aInstance, void *aShortcut );
    //     static void  UIMenuItem_SetTextColor( void *aInstance, math::vec4 *aTextColor );
    //     static void  UIMenuItem_OnTrigger( void *aInstance, void *aDelegate );
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

    //   public:
    //     static void *UIMenuSeparator_Create();
    //     static void  UIMenuSeparator_Destroy( void *aInstance );
    };

    class UIMenu : public UIMenuItem
    {
      public:
        UIMenu()                 = default;
        UIMenu( UIMenu const & ) = default;
        UIMenu( std::string const &aText );

        Ref<UIMenuItem> AddAction( std::string const &aText, std::string const &aShortcut );
        Ref<UIMenu>     AddMenu( std::string const &aText );
        Ref<UIMenuItem> AddSeparator();

        void Update();

      public:
        UIMenuItem *AddActionRaw( std::string const &aText, std::string const &aShortcut );
        UIMenu     *AddMenuRaw( std::string const &aText );
        UIMenuItem *AddSeparatorRaw();

      protected:
        std::vector<UIMenuItem *> mActions;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

    //   public:
    //     static void *UIMenu_Create();
    //     static void *UIMenu_CreateWithText( void *aText );
    //     static void  UIMenu_Destroy( void *aInstance );
    //     static void *UIMenu_AddAction( void *aInstance, void *aText, void *aShortcut );
    //     static void *UIMenu_AddMenu( void *aInstance, void *aText );
    //     static void *UIMenu_AddSeparator( void *aInstance );
    //     static void  UIMenu_Update( void *aInstance );
    };
} // namespace SE::Core