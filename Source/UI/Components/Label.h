#pragma once

#include "Component.h"

namespace SE::Core
{
    class UILabel : public UIComponent
    {
      public:
        UILabel() = default;

        UILabel( std::string const &aText );

        UILabel &SetText( std::string const &aText );

      private:
        std::string mText;

      private:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core