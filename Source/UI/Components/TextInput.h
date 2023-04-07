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

        std::string &GetText();
        void         SetBuffersize( uint32_t aSize );
        void         OnTextChanged( std::function<void( std::string aText )> aOnTextChanged );

      protected:
        std::string mHintText;
        ImVec4      mTextColor;
        uint32_t    mBufferSize = 0;
        std::string mBuffer;

        std::function<void( std::string aText )> mOnTextChanged;

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