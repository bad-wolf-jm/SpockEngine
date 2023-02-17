#include "Component.h"

namespace SE::Core
{
    void UIComponent::SetMargin( float aAll ) { return SetMargin( aAll, aAll, aAll, aAll ); }

    void UIComponent::SetMargin( float aTopBottom, float aLeftRight )
    {
        return SetMargin( aTopBottom, aTopBottom, aLeftRight, aLeftRight );
    }

    void UIComponent::SetMargin( float aTop, float aBottom, float aLeft, float aRight )
    {
        mMargin = math::vec4{ aTop, aBottom, aLeft, aRight };
    }

    void UIComponent::SetPadding( float aAll ) { return SetPadding( aAll, aAll, aAll, aAll ); }

    void UIComponent::SetPadding( float aTopBottom, float aLeftRight )
    {
        return SetPadding( aTopBottom, aTopBottom, aLeftRight, aLeftRight );
    }

    void UIComponent::SetPadding( float aTop, float aBottom, float aLeft, float aRight )
    {
        mPadding = math::vec4{ aTop, aBottom, aLeft, aRight };
    }

    void UIComponent::SetBorderThickness( float aAll ) { return SetBorderThickness( aAll, aAll, aAll, aAll ); }

    void UIComponent::SetBorderThickness( float aTopBottom, float aLeftRight )
    {
        return SetBorderThickness( aTopBottom, aTopBottom, aLeftRight, aLeftRight );
    }

    void UIComponent::SetBorderThickness( float aTop, float aBottom, float aLeft, float aRight )
    {
        mBorderThickness = math::vec4{ aTop, aBottom, aLeft, aRight };
    }

    void UIComponent::SetBorderColor( math::vec4 aAll ) { return SetBorderColor( aAll, aAll, aAll, aAll ); }

    void UIComponent::SetBorderColor( math::vec4 aTopBottom, math::vec4 aLeftRight )
    {
        return SetBorderColor( aTopBottom, aTopBottom, aLeftRight, aLeftRight );
    }

    void UIComponent::SetBorderColor( math::vec4 aTop, math::vec4 aBottom, math::vec4 aLeft, math::vec4 aRight )
    {
        mBorderColor[0] = aTop;
        mBorderColor[1] = aBottom;
        mBorderColor[2] = aLeft;
        mBorderColor[3] = aRight;
    }

    void UIComponent::SetBackgroundColor( math::vec4 aColor ) { mBackgroundColor = aColor; }

    void UIComponent::SetBorderRadius( float aAll ) { return SetBorderRadius( aAll, aAll, aAll, aAll ); }

    void UIComponent::SetBorderRadius( float aTopBottom, float aLeftRight )
    {
        return SetBorderRadius( aTopBottom, aTopBottom, aLeftRight, aLeftRight );
    }

    void UIComponent::SetBorderRadius( float aTop, float aBottom, float aLeft, float aRight )
    {
        mBorderRadius = math::vec4{ aTop, aBottom, aLeft, aRight };
    }

    ImVec2 UIComponent::RequiredSize()
    {
        float lWidth  = ( mMargin.z + mMargin.w ) + ( mPadding.z + mPadding.w ) + ( mBorderThickness.z + mBorderThickness.w );
        float lHeight = ( mMargin.x + mMargin.y ) + ( mPadding.x + mPadding.y ) + ( mBorderThickness.x + mBorderThickness.y );

        return ImVec2{ lWidth, lHeight };
    }

    float UIComponent::GetContentOffsetX() { return mMargin.z + mPadding.z + mBorderThickness.z; }
    float UIComponent::GetContentOffsetY() { return mMargin.x + mPadding.x + mBorderThickness.x; }

    ImVec2 UIComponent::GetContentOffset() { return ImVec2{ GetContentOffsetX(), GetContentOffsetY() }; }

    void UIComponent::DrawBackground()
    {
        //
    }

    void UIComponent::DrawBorder()
    {
        //
    }

    void UIComponent::Update( ImVec2 aPosition, ImVec2 aSize )
    {
        if( !mIsVisible ) return;

        ImGui::PushID( (void *)this );

        PushStyles();

        ImVec2 lContentSize     = aSize - UIComponent::RequiredSize();
        ImVec2 lContentPosition = aPosition + GetContentOffset();

        DrawBackground();
        DrawBorder();
        DrawContent( lContentPosition, lContentSize );
        PopStyles();

        ImGui::PopID();

        if( !mAllowDragDrop || !mIsEnabled ) return;
    }

    bool IsHovered() { return ImGui::IsItemHovered(); }

} // namespace SE::Core
