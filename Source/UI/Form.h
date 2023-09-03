#pragma once

#include "Components/Component.h"

namespace SE::Core
{
    class UIForm : public UIComponent
    {
      public:
        UIForm()                       = default;
        UIForm( UIForm const &UIForm ) = default;

        UIForm( string_t const &aTitle );

        void SetTitle( string_t const &aText );
        void SetContent( UIComponent *aContent );
        void SetSize( float aWidth, float aHeight );

        void Update();

      public:
        static void *UIForm_Create();
        static void  UIForm_Destroy( void *aInstance );
        static void  UIForm_SetTitle( void *aInstance, void *aTitle );
        static void  UIForm_SetContent( void *aInstance, void *aContent );
        static void  UIForm_Update( void *aInstance );
        static void  UIForm_SetSize( void *aInstance, float aWidth, float aHeight );

      protected:
        string_t  mTitle;
        UIComponent *mContent = nullptr;

        float mWidth         = 0.0f;
        float mHeight        = 0.0f;
        bool  mResizeRequest = false;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core