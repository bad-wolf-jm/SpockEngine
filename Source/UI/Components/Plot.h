#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIPlot : public UIComponent
    {
      public:
        UIPlot() = default;

        UIPlot( std::string const &aText );

        void SetText( std::string const &aText );

      protected:
        std::string mText;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core