#pragma once

#include "Component.h"

namespace SE::Core
{
    class UILabel : public UIComponent
    {
      public:
        UILabel() = default;

        UILabel( string_t const &aText );

        void SetText( string_t const &aText );
        void SetTextColor( math::vec4 aColor );

        ImVec2 RequiredSize();

      protected:
        string_t mText;
        ImVec4   mTextColor;

      protected:
        void PushStyles();
        void PopStyles();

        void DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core