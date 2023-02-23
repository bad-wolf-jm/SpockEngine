#include "Component.h"

namespace SE::Core
{
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

    ImVec2 UIComponent::RequiredSize()
    {
        float lWidth  = ( mPadding.z + mPadding.w );
        float lHeight = ( mPadding.x + mPadding.y );

        return ImVec2{ lWidth, lHeight };
    }

    ImVec2 UIComponent::GetContentPadding()
    {
        float lWidth  = ( mPadding.z + mPadding.w );
        float lHeight = ( mPadding.x + mPadding.y );

        return ImVec2{ lWidth, lHeight };
    }

    float UIComponent::GetContentOffsetX() { return mPadding.z; }
    float UIComponent::GetContentOffsetY() { return mPadding.x; }

    ImVec2 UIComponent::GetContentOffset() { return ImVec2{ GetContentOffsetX(), GetContentOffsetY() }; }

    ImVec2 UIComponent::GetContentAlignedposition( eHorizontalAlignment const &aHAlignment, eVerticalAlignment const &aVAlignment,
                                                   ImVec2 aPosition, ImVec2 aContentSize, ImVec2 aSize )
    {
        ImVec2 lContentPosition{};
        switch( aHAlignment )
        {
        case eHorizontalAlignment::LEFT: lContentPosition.x = aPosition.x; break;
        case eHorizontalAlignment::RIGHT: lContentPosition.x = aPosition.x + ( aSize.x - aContentSize.x ); break;
        case eHorizontalAlignment::CENTER:
        default: lContentPosition.x = aPosition.x + ( aSize.x - aContentSize.x ) / 2.0f; break;
        }

        switch( aVAlignment )
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

        ImVec2 lContentSize     = aSize - GetContentPadding();
        ImVec2 lContentPosition = aPosition + GetContentOffset();

        DrawContent( lContentPosition, lContentSize );

        PopStyles();

        ImGui::PopID();

        if( !mAllowDragDrop || !mIsEnabled ) return;
    }

    bool IsHovered() { return ImGui::IsItemHovered(); }

} // namespace SE::Core
