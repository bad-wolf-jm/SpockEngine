#pragma once

#include "Component.h"

namespace SE::Core
{
    class UITextInput : public UIComponent
    {
      public:
        UITextInput() = default;

        UITextInput( string_t const &aHintText );

        void SetHintText( string_t const &aHintText );
        void SetTextColor( math::vec4 aColor );

        string_t &GetText();
        void         SetBuffersize( uint32_t aSize );
        void         OnTextChanged( std::function<void( string_t aText )> aOnTextChanged );

      protected:
        string_t mHintText;
        ImVec4      mTextColor;
        uint32_t    mBufferSize = 0;
        string_t mBuffer;

        std::function<void( string_t aText )> mOnTextChanged;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      private:
        void *mOnTextChangedDelegate       = nullptr;
        int   mOnTextChangedDelegateHandle = -1;

      public:
        static void *UITextInput_Create();
        static void *UITextInput_CreateWithText( void *aText );
        static void  UITextInput_Destroy( void *aInstance );
        static void *UITextInput_GetText( void *aInstance );
        static void  UITextInput_SetHintText( void *aInstance, void *aText );
        static void  UITextInput_SetTextColor( void *aInstance, math::vec4 *aTextColor );
        static void  UITextInput_SetBufferSize( void *aInstance, uint32_t aNewSize );
        static void  UITextInput_OnTextChanged( void *aInstance, void *aDelegate );
    };
} // namespace SE::Core