#pragma once

#include "Component.h"

namespace SE::Core
{
    class UISlider : public UIComponent
    {
      public:
        UISlider() = default;

        ImVec2 RequiredSize();

      protected:
        float mValue;

      protected:
        void PushStyles();
        void PopStyles();

        void DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core