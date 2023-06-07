#include "Component.h"

#include "Engine/Engine.h"

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

    void UIComponent::SetBackgroundColor( math::vec4 aColor ) { mBackgroundColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }

    void UIComponent::SetFont( FontFamilyFlags aFont ) { mFont = aFont; }
    void UIComponent::SetTooltip(UIComponent* aToolTip) { mTooltip = aToolTip; }

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
        default: lContentPosition.x = aPosition.x + ( aSize.x - aContentSize.x ) * 0.5f; break;
        }

        switch( aVAlignment )
        {
        case eVerticalAlignment::TOP: lContentPosition.y = aPosition.y; break;
        case eVerticalAlignment::BOTTOM: lContentPosition.y = aPosition.y + ( aSize.y - aContentSize.y ); break;
        case eVerticalAlignment::CENTER:
        default: lContentPosition.y = aPosition.y + ( aSize.y - aContentSize.y ) * 0.5f; break;
        }

        return lContentPosition;
    }

    void UIComponent::Update( ImVec2 aPosition, ImVec2 aSize )
    {
        if( !mIsVisible ) return;

        ImGui::PushID( (void *)this );

        ImGui::SetCursorPos( aPosition );
        if( ( mBackgroundColor.x != 0.0f ) || ( mBackgroundColor.y != 0.0f ) || ( mBackgroundColor.z != 0.0f ) ||
            ( mBackgroundColor.w != 0.0f ) )
        {
            auto lDrawList       = ImGui::GetWindowDrawList();
            auto lScreenPosition = ImGui::GetCursorScreenPos();
            lDrawList->AddRectFilled( lScreenPosition, lScreenPosition + aSize, ImColor( mBackgroundColor ), 0.0f );
        }

        SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( mFont );
        PushStyles();

        ImVec2 lContentSize     = aSize - GetContentPadding();
        ImVec2 lContentPosition = aPosition + GetContentOffset();

        DrawContent( lContentPosition, lContentSize );
        if( ImGui::IsItemHovered() && (mTooltip != nullptr))
        {
            ImGui::BeginTooltip();
            mTooltip->Update(ImVec2{}, mTooltip->RequiredSize());
            ImGui::EndTooltip();
        }

        PopStyles();
        SE::Core::Engine::GetInstance()->UIContext()->PopFont();

        ImGui::PopID();

        if( !mAllowDragDrop || !mIsEnabled ) return;
    }

    bool IsHovered() { return ImGui::IsItemHovered(); }

    // void UIComponent::UIComponent_SetIsVisible( void *aSelf, bool aIsVisible )
    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );

    //     lSelf->mIsVisible = aIsVisible;
    // }

    // void UIComponent::UIComponent_SetIsEnabled( void *aSelf, bool aIsEnabled )
    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );

    //     lSelf->mIsEnabled = aIsEnabled;
    // }

    // void UIComponent::UIComponent_SetAllowDragDrop( void *aSelf, bool aAllowDragDrop )
    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );

    //     lSelf->mAllowDragDrop = aAllowDragDrop;
    // }

    // void UIComponent::UIComponent_SetPaddingAll( void *aSelf, float aPaddingAll )
    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );

    //     lSelf->SetPadding( aPaddingAll );
    // }

    // void UIComponent::UIComponent_SetPaddingPairs( void *aSelf, float aPaddingTopBottom, float aPaddingLeftRight )
    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );

    //     lSelf->SetPadding( aPaddingTopBottom, aPaddingLeftRight );
    // }

    // void UIComponent::UIComponent_SetPaddingIndividual( void *aSelf, float aPaddingTop, float aPaddingBottom, float aPaddingLeft,
    //                                                     float aPaddingRight )

    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );

    //     lSelf->SetPadding( aPaddingTop, aPaddingBottom, aPaddingLeft, aPaddingRight );
    // }

    // void UIComponent::UIComponent_SetAlignment( void *aSelf, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );

    //     lSelf->SetAlignment( aHAlignment, aVAlignment );
    // }

    // void UIComponent::UIComponent_SetHorizontalAlignment( void *aSelf, eHorizontalAlignment aAlignment )
    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );

    //     lSelf->SetHorizontalAlignment( aAlignment );
    // }

    // void UIComponent::UIComponent_SetVerticalAlignment( void *aSelf, eVerticalAlignment aAlignment )
    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );

    //     lSelf->SetVerticalAlignment( aAlignment );
    // }

    // void UIComponent::UIComponent_SetBackgroundColor( void *aSelf, math::vec4 aColor )
    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );

    //     lSelf->SetBackgroundColor( aColor );
    // }

    // void UIComponent::UIComponent_SetFont( void *aSelf, FontFamilyFlags aFont )
    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );

    //     lSelf->SetFont( aFont );
    // }

    // void UIComponent::UIComponent_SetTooltip( void *aSelf, void *aTooltip )
    // {
    //     auto lSelf = static_cast<UIComponent *>( aSelf );
    //     auto lTooltip = static_cast<UIComponent *>( aTooltip );

    //     lSelf->SetTooltip( lTooltip );
    // }

} // namespace SE::Core
