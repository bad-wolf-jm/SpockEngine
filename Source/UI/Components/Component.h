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

        void SetMargin( float aMarginAll );
        void SetMargin( float aMarginTopBottom, float aMarginLeftRight );
        void SetMargin( float aMarginTop, float aMarginBottom, float aMarginLeft, float aMarginRight );

        void SetPadding( float aPaddingAll );
        void SetPadding( float aPaddingTopBottom, float aPaddingLeftRight );
        void SetPadding( float aPaddingTop, float aPaddingBottom, float aPaddingLeft, float aPaddingRight );

        void SetBorderColor( math::vec4 aBorderColor );
        void SetBorderColor( math::vec4 aTopBottom, math::vec4 aLeftRight );
        void SetBorderColor( math::vec4 aTop, math::vec4 aBottom, math::vec4 aLeft, math::vec4 aRight );

        void SetBackgroundColor( math::vec4 aColor );

        void SetBorderThickness( float aPaddingAll );
        void SetBorderThickness( float aPaddingTopBottom, float aPaddingLeftRight );
        void SetBorderThickness( float aPaddingTop, float aPaddingBottom, float aPaddingLeft, float aPaddingRight );

        void SetBorderRadius( float aPaddingAll );
        void SetBorderRadius( float aPaddingTopBottom, float aPaddingLeftRight );
        void SetBorderRadius( float aPaddingTop, float aPaddingBottom, float aPaddingLeft, float aPaddingRight );

      protected:
        math::vec4 mMargin{};
        math::vec4 mPadding{};
        math::vec4 mBorderThickness{};
        math::vec4 mBorderRadius{};
        math::vec4 mBackgroundColor{};

        std::array<math::vec4, 4> mBorderColor{};

      protected:
        virtual void PushStyles() = 0;
        virtual void PopStyles()  = 0;

        float  GetContentOffsetX();
        float  GetContentOffsetY();
        ImVec2 GetContentOffset();

        void DrawBackground();
        void DrawBorder();

        virtual void DrawContent( ImVec2 aPosition, ImVec2 aSize ) = 0;

        bool IsHovered();
    };
} // namespace SE::Core