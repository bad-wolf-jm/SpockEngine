#pragma once

#include "Components/Component.h"

namespace SE::Core
{
    class UIForm : public UIComponent
    {
      public:
        UIForm()                       = default;
        UIForm( UIForm const &UIForm ) = default;

        UIForm( std::string const &aTitle );

        void SetTitle( std::string const &aText );
        void SetContent( UIComponent *aContent );

        void Update();

      public:
        static void *UIForm_Create();
        static void  UIForm_Destroy( void *aInstance );
        static void  UIForm_SetTitle( void *aInstance, void *aTitle );
        static void  UIForm_SetContent( void *aInstance, void *aContent );
        static void  UIForm_Update( void *aInstance );

      protected:
        std::string  mTitle;
        UIComponent *mContent = nullptr;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core