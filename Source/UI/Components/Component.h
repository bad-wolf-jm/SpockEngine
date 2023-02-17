#pragma once

#include "UI/UI.h"

namespace SE::Core
{
    class UIComponent
    {
      public:
        bool mIsVisible     = true;
        bool mIsEnabled     = true;
        bool mAllowDragDrop = true;

      public:
        UIComponent()  = default;
        ~UIComponent() = default;

        void Update( ImVec2 aPosition, ImVec2 aSize );

        virtual ImVec2 RequiredSize() = 0;

      protected:
        virtual void PushStyles()                                  = 0;
        virtual void PopStyles()                                   = 0;
        virtual void DrawContent( ImVec2 aPosition, ImVec2 aSize ) = 0;

        bool IsHovered();
    };
} // namespace SE::Core