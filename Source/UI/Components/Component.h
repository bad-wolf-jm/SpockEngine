#pragma once

#include "UI/UI.h"
#include "UI/UIContext.h"

namespace SE::Core
{
    enum class eHorizontalAlignment : int32_t
    {
        LEFT,
        RIGHT,
        CENTER
    };

    enum class eVerticalAlignment : int32_t
    {
        TOP,
        BOTTOM,
        CENTER
    };

    class UIComponent
    {
      public:
        bool            mIsVisible     = true;
        bool            mIsEnabled     = true;
        bool            mAllowDragDrop = true;
        FontFamilyFlags mFont          = FontFamilyFlags::NORMAL;

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

        void   SetBackgroundColor( math::vec4 aColor );
        ImVec4 BackgroundColor()
        {
            return mBackgroundColor;
        }

        void SetFont( FontFamilyFlags aFont );

        void SetTooltip( UIComponent *aToolTip );

      protected:
        math::vec4 mPadding{};
        ImVec4     mBackgroundColor{};

        eHorizontalAlignment mHAlign = eHorizontalAlignment::CENTER;
        eVerticalAlignment   mVAlign = eVerticalAlignment::CENTER;

        UIComponent *mTooltip = nullptr;

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

      public:
        static void UIComponent_SetIsVisible( void *aSelf, bool aIsVisible );
        static void UIComponent_SetIsEnabled( void *aSelf, bool aIsEnabled );
        static void UIComponent_SetAllowDragDrop( void *aSelf, bool aAllowDragDrop );

        static void UIComponent_SetPaddingAll( void *aSelf, float aPaddingAll );
        static void UIComponent_SetPaddingPairs( void *aSelf, float aPaddingTopBottom, float aPaddingLeftRight );
        static void UIComponent_SetPaddingIndividual( void *aSelf, float aPaddingTop, float aPaddingBottom, float aPaddingLeft,
                                                      float aPaddingRight );
        static void UIComponent_SetAlignment( void *aSelf, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment );
        static void UIComponent_SetHorizontalAlignment( void *aSelf, eHorizontalAlignment aAlignment );
        static void UIComponent_SetVerticalAlignment( void *aSelf, eVerticalAlignment aAlignment );
        static void UIComponent_SetBackgroundColor( void *aSelf, math::vec4 aColor );
        static void UIComponent_SetFont( void *aSelf, FontFamilyFlags aFont );
        static void UIComponent_SetTooltip( void *aSelf, void *aTooltip );
    };
} // namespace SE::Core