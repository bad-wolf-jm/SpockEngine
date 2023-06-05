#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIVector3Input : public UIComponent
    {
      public:
        UIVector3Input() = default;

        UIVector3Input( std::string const &aText );

        void SetText( std::string const &aText );
        void SetTextColor( math::vec4 aColor );

        ImVec2 RequiredSize();
        
      protected:
        std::string mText;
        ImVec4      mTextColor;

      protected:
        void PushStyles();
        void PopStyles();

        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UILabel_Create();
        static void *UILabel_CreateWithText( void *aText );
        static void  UILabel_Destroy( void *aInstance );
        static void  UILabel_SetText( void *aInstance, void *aText );
        static void  UILabel_SetTextColor( void *aInstance, math::vec4 aTextColor );
    };
} // namespace SE::Core