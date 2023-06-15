#pragma once

#include "Component.h"
#include "Label.h"

namespace SE::Core
{
    class UIButton : public UILabel
    {
      public:
        UIButton() = default;

        UIButton( string_t const &aText );
        UIButton( string_t const &aText, std::function<void()> aOnClick );

        void OnClick( std::function<void()> aOnClick );
        void SetText( string_t const &aText );

      private:
        std::function<void()> mOnClick;

      private:
        void PushStyles();
        void PopStyles();

        void PushStyles( bool aEnabled );
        void PopStyles( bool aEnabled );

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

    //   public:
        // void ( *mOnClickDelegate )() = nullptr;
    };
} // namespace SE::Core