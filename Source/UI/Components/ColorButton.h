#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIColorButton : public UIComponent
    {
      public:
        UIColorButton() = default;

        ImVec2 RequiredSize();

      protected:
        ImVec4 mColor;

      protected:
        void PushStyles();
        void PopStyles();

        void DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core