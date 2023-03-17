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
        void SetBackgroundColor( math::vec4 aColor );

      protected:
        std::string mText;
        ImVec4      mBackgroundColor;
        ImVec4      mTextColor;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core