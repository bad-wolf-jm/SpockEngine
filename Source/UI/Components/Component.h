#pragma once

#include "UI/UI.h"

namespace SE::Core
{
    enum class eHorizontalAlignment : uint8_t
    {
        LEFT,
        RIGHT,
        CENTER
    };

    enum class eVerticalAlignment : uint8_t
    {
        TOP,
        BOTTOM,
        CENTER
    };

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

        void SetPadding( float aPaddingAll );
        void SetPadding( float aPaddingTopBottom, float aPaddingLeftRight );
        void SetPadding( float aPaddingTop, float aPaddingBottom, float aPaddingLeft, float aPaddingRight );

        void SetAlignment( eHorizontalAlignment const &aHAlignment, eVerticalAlignment const &aVAlignment );
        void SetHorizontalAlignment( eHorizontalAlignment const &aAlignment );
        void SetVerticalAlignment( eVerticalAlignment const &aAlignment );

        void SetBackgroundColor( math::vec4 aColor );

      protected:
        math::vec4 mPadding{};
        ImVec4     mBackgroundColor{};

        eHorizontalAlignment mHAlign = eHorizontalAlignment::CENTER;
        eVerticalAlignment   mVAlign = eVerticalAlignment::CENTER;

      protected:
        virtual void PushStyles() = 0;
        virtual void PopStyles()  = 0;

        float  GetContentOffsetX();
        float  GetContentOffsetY();
        ImVec2 GetContentOffset();
        ImVec2 GetContentPadding();

        ImVec2 GetContentAlignedposition( eHorizontalAlignment const &aHAlignment, eVerticalAlignment const &aVAlignment,
                                          ImVec2 aPosition, ImVec2 aContentSize, ImVec2 aSize );

        virtual void DrawContent( ImVec2 aPosition, ImVec2 aSize ) = 0;

        bool IsHovered();
    };
} // namespace SE::Core