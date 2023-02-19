#include "Component.h"

namespace SE::Core
{
    // void UIComponent::SetMargin( float aAll ) { return SetMargin( aAll, aAll, aAll, aAll ); }

    // void UIComponent::SetMargin( float aTopBottom, float aLeftRight )
    // {
    //     return SetMargin( aTopBottom, aTopBottom, aLeftRight, aLeftRight );
    // }

    // void UIComponent::SetMargin( float aTop, float aBottom, float aLeft, float aRight )
    // {
    //     mMargin = math::vec4{ aTop, aBottom, aLeft, aRight };
    // }

    void UIComponent::SetPadding( float aAll ) { return SetPadding( aAll, aAll, aAll, aAll ); }

    void UIComponent::SetPadding( float aTopBottom, float aLeftRight )
    {
        return SetPadding( aTopBottom, aTopBottom, aLeftRight, aLeftRight );
    }


    void UIComponent::SetAlignment( eHorizontalAlignment const &aHAlignment, eVerticalAlignment const &aVAlignment )
    {
        SetHorizontalAlignment( aHAlignment );
        SetVerticalAlignment( aVAlignment );
    }

    void UIComponent::SetHorizontalAlignment( eHorizontalAlignment const &aAlignment ) { mHAlign = aAlignment; }
    void UIComponent::SetVerticalAlignment( eVerticalAlignment const &aAlignment ) { mVAlign = aAlignment; }



    void UIComponent::SetPadding( float aTop, float aBottom, float aLeft, float aRight )
    {
        mPadding = math::vec4{ aTop, aBottom, aLeft, aRight };
    }

    // void UIComponent::SetBorderThickness( float aAll ) { return SetBorderThickness( aAll, aAll, aAll, aAll ); }

    // void UIComponent::SetBorderThickness( float aTopBottom, float aLeftRight )
    // {
    //     return SetBorderThickness( aTopBottom, aTopBottom, aLeftRight, aLeftRight );
    // }

    // void UIComponent::SetBorderThickness( float aTop, float aBottom, float aLeft, float aRight )
    // {
    //     mBorderThickness = math::vec4{ aTop, aBottom, aLeft, aRight };
    // }

    // void UIComponent::SetBorderColor( math::vec4 aAll ) { return SetBorderColor( aAll, aAll, aAll, aAll ); }

    // void UIComponent::SetBorderColor( math::vec4 aTopBottom, math::vec4 aLeftRight )
    // {
    //     return SetBorderColor( aTopBottom, aTopBottom, aLeftRight, aLeftRight );
    // }

    // void UIComponent::SetBorderColor( math::vec4 aTop, math::vec4 aBottom, math::vec4 aLeft, math::vec4 aRight )
    // {
    //     mBorderColor[0] = aTop;
    //     mBorderColor[1] = aBottom;
    //     mBorderColor[2] = aLeft;
    //     mBorderColor[3] = aRight;
    // }

    // void UIComponent::SetBackgroundColor( math::vec4 aColor ) { mBackgroundColor = aColor; }

    // void UIComponent::SetBorderRadius( float aAll ) { return SetBorderRadius( aAll, aAll, aAll, aAll ); }

    // void UIComponent::SetBorderRadius( float aTopBottom, float aLeftRight )
    // {
    //     return SetBorderRadius( aTopBottom, aTopBottom, aLeftRight, aLeftRight );
    // }

    // void UIComponent::SetBorderRadius( float aTop, float aBottom, float aLeft, float aRight )
    // {
    //     mBorderRadius = math::vec4{ aTop, aBottom, aLeft, aRight };
    // }

    ImVec2 UIComponent::RequiredSize()
    {
        float lWidth  = ( mPadding.z + mPadding.w );
        float lHeight = ( mPadding.x + mPadding.y );

        return ImVec2{ lWidth, lHeight };
    }

    float UIComponent::GetContentOffsetX() { return mPadding.z; }
    float UIComponent::GetContentOffsetY() { return mPadding.x; }

    ImVec2 UIComponent::GetContentOffset() { return ImVec2{ GetContentOffsetX(), GetContentOffsetY() }; }

    // void UIComponent::DrawBackground( ImVec2 aPosition, ImVec2 aSize )
    // {
    //     auto lDrawList = ImGui::GetWindowDrawList();

    //     ImVec2 lTopLeft     = ImGui::GetCursorScreenPos() + aPosition + ImVec2( 0.50f, 0.50f );
    //     ImVec2 lBottomRight = ImGui::GetCursorScreenPos() + ( aPosition + aSize ) - ImVec2( 0.50f, 0.50f );

    //     if( mBorderRadius == math::vec4( 0.0f ) )
    //     {
    //         lDrawList->PathLineTo( lTopLeft );
    //         lDrawList->PathLineTo( ImVec2( lBottomRight.x, lTopLeft.y ) );
    //         lDrawList->PathLineTo( lBottomRight );
    //         lDrawList->PathLineTo( ImVec2( lTopLeft.x, lBottomRight.y ) );
    //     }
    //     else
    //     {
    //         lDrawList->PathArcToFast( ImVec2( lTopLeft.x + mBorderRadius.x, lTopLeft.y + mBorderRadius.x ), mBorderRadius.x, 6, 9 );
    //         lDrawList->PathArcToFast( ImVec2( lBottomRight.x - mBorderRadius.y, lTopLeft.y + mBorderRadius.y ), mBorderRadius.y, 9, 12 );
    //         lDrawList->PathArcToFast( ImVec2( lBottomRight.x - mBorderRadius.z, lBottomRight.y - mBorderRadius.z ), mBorderRadius.z, 0, 3 );
    //         lDrawList->PathArcToFast( ImVec2( lTopLeft.x + mBorderRadius.w, lBottomRight.y - mBorderRadius.w ), mBorderRadius.w, 3, 6 );
    //     }

    //     // PathStroke( IM_COL32( 255, 255, 255, 255 ), ImDrawFlags_Closed, mBorderThickness.x );
    //     lDrawList->PathFillConvex(IM_COL32( 255, 255, 255, 255 ));
    // }

    // void UIComponent::DrawBorder(ImVec2 aPosition, ImVec2 aSize)
    // {
    //     //
    // }

    ImVec2 UIComponent::GetContentAlignedposition(ImVec2 aPosition, ImVec2 aContentSize, ImVec2 aSize)
    {
        ImVec2 lContentPosition{};
        switch( mHAlign )
        {
        case eHorizontalAlignment::LEFT: lContentPosition.x = aPosition.x; break;
        case eHorizontalAlignment::RIGHT: lContentPosition.x = aPosition.x + ( aSize.x - aContentSize.x ); break;
        case eHorizontalAlignment::CENTER:
        default: lContentPosition.x = aPosition.x + ( aSize.x - aContentSize.x ) / 2.0f; break;
        }

        switch( mVAlign )
        {
        case eVerticalAlignment::TOP: lContentPosition.y = aPosition.y; break;
        case eVerticalAlignment::BOTTOM: lContentPosition.y = aPosition.y + ( aSize.y - aContentSize.y ); break;
        case eVerticalAlignment::CENTER:
        default: lContentPosition.y = aPosition.y + ( aSize.y - aContentSize.y ) / 2.0f; break;
        }

        return lContentPosition;

    }

    void UIComponent::Update( ImVec2 aPosition, ImVec2 aSize )
    {
        if( !mIsVisible ) return;

        ImGui::PushID( (void *)this );

        PushStyles();

        ImVec2 lContentSize     = aSize - UIComponent::RequiredSize();
        ImVec2 lContentPosition = aPosition + GetContentOffset();

        // DrawBackground(aPosition, aSize);
        // DrawBorder(aPosition, aSize);

        DrawContent( lContentPosition, lContentSize );

        PopStyles();

        ImGui::PopID();

        if( !mAllowDragDrop || !mIsEnabled ) return;
    }

    bool IsHovered() { return ImGui::IsItemHovered(); }

} // namespace SE::Core
