#pragma once

#include "Components/Component.h"

namespace SE::Core
{
    class UIForm : public UIComponent
    {
      public:
        UIForm()                       = default;
        UIForm( UIForm const &UIForm ) = default;

        UIForm( std::string const &aTitle );

        void SetTitle( std::string const &aText );
        void SetContent( UIComponent *aContent );

        void Update();

      protected:
        std::string  mTitle;
        UIComponent *mContent;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core