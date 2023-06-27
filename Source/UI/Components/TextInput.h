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
        void      SetBuffersize( uint32_t aSize );
        void      OnTextChanged( std::function<void( string_t aText )> aOnTextChanged );

      protected:
        string_t mHintText;
        ImVec4   mTextColor;
        uint32_t mBufferSize = 0;
        string_t mBuffer;

        std::function<void( string_t aText )> mOnTextChanged;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        void *mOnTextChangedDelegate       = nullptr;
        int   mOnTextChangedDelegateHandle = -1;
    };
} // namespace SE::Core