#pragma once

#include "Component.h"
#include "Label.h"

namespace SE::Core
{
    class UITextToggleButton : public UILabel
    {
      public:
        UITextToggleButton() = default;

        UITextToggleButton( std::string const &aText );
        UITextToggleButton( std::string const &aText, std::function<void( bool )> aOnClick );

        void OnChange( std::function<void( bool )> aOnClick );

        void SetText( std::string const &aText );
        void SetActiveColor( math::vec4 const &aColor );
        void SetInactiveColor( math::vec4 const &aColor );

      private:
        std::function<void(bool)> mOnChange;

        bool   mActivated     = false;
        ImVec4 mActiveColor   = ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f };
        ImVec4 mInactiveColor = ImVec4{ 1.0f, 1.0f, 1.0f, .2f };

      private:
        void PushStyles();
        void PopStyles();

        void PushStyles( bool aEnabled );
        void PopStyles( bool aEnabled );

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core