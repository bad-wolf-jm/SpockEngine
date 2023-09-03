#pragma once

#include "Components/Component.h"

#include <Core/Math/Types.h>

namespace SE::Core
{
    class UIDialog : public UIComponent
    {
      public:
        UIDialog();
        ~UIDialog() = default;

        UIDialog( UIDialog const & ) = default;
        UIDialog( string_t aTitle, math::vec2 aSize );

        void SetTitle( string_t const &aText );
        void SetSize( math::vec2 aSize );
        void SetContent( UIComponent *aContent );

        void Open();
        void Close();

        void Update();

      public:
        static void *UIDialog_Create();
        static void *UIDialog_CreateWithTitleAndSize( void *aTitle, math::vec2 *aSize );
        static void  UIDialog_Destroy( void *aInstance );
        static void  UIDialog_SetTitle( void *aInstance, void *aTitle );
        static void  UIDialog_SetSize( void *aInstance, math::vec2 aSize );
        static void  UIDialog_SetContent( void *aInstance, void *aContent );
        static void  UIDialog_Open( void *aInstance );
        static void  UIDialog_Close( void *aInstance );
        static void  UIDialog_Update( void *aInstance );

      protected:
        string_t  mTitle;
        UIComponent *mContent = nullptr;
        math::vec2   mSize;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };

} // namespace SE::Core