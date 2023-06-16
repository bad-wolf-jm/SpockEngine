#pragma once

#include "UI/Components/Component.h"

namespace SE::Core
{
    class UIContainer : public UIComponent
    {
      public:
        UIContainer() = default;

        UIContainer( UIContainer const & ) = default;

        ~UIContainer() = default;

        void SetContent( UIComponent *aChild );

        ImVec2 RequiredSize();

      protected:
        UIComponent *mContent;

      protected:
        void PushStyles();
        void PopStyles();

        void DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core