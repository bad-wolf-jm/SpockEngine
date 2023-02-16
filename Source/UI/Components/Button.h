#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIButton : public UIComponent
    {
      public:
        UIButton( std::string const &aText );
        UIButton( std::string const &aText, std::function<void()> aOnClick );

      private:
        std::string           mText;
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