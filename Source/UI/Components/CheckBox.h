#pragma once

#include "Component.h"
#include "Label.h"

namespace SE::Core
{
    class UICheckBox : public UIComponent
    {
      public:
        UICheckBox() = default;
        UICheckBox( std::function<void()> aOnClick );

        void OnClick( std::function<void()> aOnClick );

      private:
        std::function<void()> mOnClick;
        bool                  mIsChecked = false;

      private:
        void PushStyles();
        void PopStyles();

        void PushStyles( bool aEnabled );
        void PopStyles( bool aEnabled );

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core