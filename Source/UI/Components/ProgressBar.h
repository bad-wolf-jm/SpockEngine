#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIProgressBar : public UIComponent
    {
      public:
        UIProgressBar() = default;

        UIProgressBar( string_t const &aText );

        void SetText( string_t const &aValue );
        void SetTextColor( math::vec4 aColor );
        void SetProgressValue( float aValue );
        void SetProgressColor( math::vec4 aColor );
        void SetThickness( float aValue );

      protected:
        string_t mText;
        ImVec4   mTextColor;
        float    mValue;
        ImVec4   mProgressColor;
        float    mThickness;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core