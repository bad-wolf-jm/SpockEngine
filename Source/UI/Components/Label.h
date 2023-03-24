#pragma once

#include "Component.h"

namespace SE::Core
{
    class UILabel : public UIComponent
    {
      public:
        UILabel() = default;

        UILabel( std::string const &aText );

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
        static void *UILabel_Create();
        static void *UILabel_CreateWithText( void *aText );
        static void  UILabel_Destroy( void *aInstance );
        static void  UILabel_SetText( void *aInstance, void *aText );
        static void  UILabel_SetTextColor( void *aInstance, math::vec4 *aTextColor );
        static void  UILabel_Update( void *aInstance, math::vec2 *aPosition, math::vec2 *aSize );
    };
} // namespace SE::Core