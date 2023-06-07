#pragma once

#include "Components/Component.h"

#include <Core/Math/Types.h>

namespace SE::Core
{
    class UIDialog : public UIComponent
    {
      public:
        UIDialog();
        ~UIDialog() = default;

        UIDialog( UIDialog const & ) = default;
        UIDialog( std::string aTitle, math::vec2 aSize );

        void SetTitle( std::string const &aText );
        void SetSize( math::vec2 aSize );
        void SetContent( UIComponent *aContent );

        void Open();
        void Close();

        void Update();

      protected:
        std::string  mTitle;
        UIComponent *mContent = nullptr;
        math::vec2   mSize;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };

} // namespace SE::Core