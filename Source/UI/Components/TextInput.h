#pragma once

#include "Component.h"

namespace SE::Core
{
    class UITextInput : public UIComponent
    {
      public:
        UITextInput() = default;

        UITextInput( std::string const &aHintText );

        void SetHintText( std::string const &aHintText );
        void SetTextColor( math::vec4 aColor );

        void SetBuffersize( uint32_t aSize );

      protected:
        std::string mHintText;
        ImVec4      mTextColor;
        uint32_t    mBufferSize = 0;
        std::string mBuffer;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UITextInput_Create();
        static void *UITextInput_CreateWithText( void *aText );
        static void  UITextInput_Destroy( void *aInstance );
        static void  UITextInput_SetHintText( void *aInstance, void *aText );
        static void  UITextInput_SetTextColor( void *aInstance, math::vec4 *aTextColor );
    };
} // namespace SE::Core