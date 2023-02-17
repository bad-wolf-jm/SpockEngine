#pragma once

#include "Component.h"
#include "Label.h"

namespace SE::Core
{
    class UIButton : public UILabel
    {
      public:
        UIButton() = default;

        UIButton( std::string const &aText );
        UIButton( std::string const &aText, std::function<void()> aOnClick );

        void OnClick( std::function<void()> aOnClick );
        void SetText( std::string const &aText );

      private:
        std::function<void()> mOnClick;

      private:
        void PushStyles();
        void PopStyles();

        void PushStyles( bool aEnabled );
        void PopStyles( bool aEnabled );

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core