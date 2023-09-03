#include "BoxLayout.h"

namespace SE::Core
{

    UIBoxLayout::UIBoxLayout( eBoxLayoutOrientation aOrientation )
        : mOrientation{ aOrientation }
    {
    }

    void UIBoxLayout::PushStyles()
    {
    }
    void UIBoxLayout::PopStyles()
    {
    }

    ImVec2 UIBoxLayout::RequiredSize()
    {
        float lWidth  = 0.0f;
        float lHeight = 0.0f;

        for( auto const &lItem : mChildren )
        {
            if( ( lItem.mItem != nullptr ) && ( !lItem.mItem->mIsVisible ) )
                continue;

            if( lItem.mIsSeparator )
            {
                if( mOrientation == eBoxLayoutOrientation::HORIZONTAL )
                {
                    lWidth += 1.0f;
                    lHeight = math::max( lHeight, 0.0f );
                }
                else
                {
                    lHeight += 1.0f;
                    lWidth = math::max( lWidth, 0.0f );
                }
                continue;
            }

            ImVec2 lRequiredSize{};
            if( lItem.mFixedSize > 0.0f )
            {
                if( lItem.mItem )
                    lRequiredSize = lItem.mItem->RequiredSize();

                if( mOrientation == eBoxLayoutOrientation::HORIZONTAL )
                {
                    lWidth += lItem.mFixedSize;
                    lHeight = math::max( lHeight, lRequiredSize.y );
                }
                else
                {
                    lHeight += lItem.mFixedSize;
                    lWidth = math::max( lWidth, lRequiredSize.x );
                }
            }
            else
            {

                if( lItem.mItem )
                    lRequiredSize = lItem.mItem->RequiredSize();

                if( mOrientation == eBoxLayoutOrientation::HORIZONTAL )
                {
                    lWidth += lRequiredSize.x;
                    lHeight = math::max( lHeight, lRequiredSize.y );
                }
                else
                {
                    lHeight += lRequiredSize.y;
                    lWidth = math::max( lWidth, lRequiredSize.x );
                }
            }
        }

        return ImVec2{ lWidth, lHeight } + UIComponent::RequiredSize();
    }

    void UIBoxLayout::SetOrientation( eBoxLayoutOrientation aValue )
    {
        mOrientation = aValue;
    }

    void UIBoxLayout::SetItemSpacing( float aItemSpacing )
    {
        mItemSpacing = aItemSpacing;
    }

    void UIBoxLayout::Add( UIComponent *aChild, bool aExpand, bool aFill )
    {
        Add( aChild, 0.0f, aExpand, aFill, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER );
    }

    void UIBoxLayout::Add( UIComponent *aChild, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                           eVerticalAlignment const &aVAlignment )
    {
        Add( aChild, 0.0f, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIBoxLayout::Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill )
    {
        Add( aChild, aFixedSize, aExpand, aFill, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER );
    }

    void UIBoxLayout::Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                           eVerticalAlignment const &aVAlignment )
    {
        mChildren.push_back( BoxLayoutItem{ aChild, aFixedSize, aExpand, aFill, false, aHAlignment, aVAlignment, ImVec4{} } );
    }

    void UIBoxLayout::AddSeparator()
    {
        mChildren.push_back( BoxLayoutItem{ nullptr, 0.0f, false, false, true, eHorizontalAlignment::CENTER,
                                            eVerticalAlignment::CENTER, ImVec4{ 1.0f, 0.0f, 0.0f, 1.0f } } );
    }

    void UIBoxLayout::Clear()
    {
        mChildren.clear();
    }

    void UIBoxLayout::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        auto     lGlobalPosition = ImGui::GetCursorScreenPos();
        uint32_t lExpandCount    = 0;

        vector_t<BoxLayoutItem> lVisibleChildren;
        std::copy_if( mChildren.begin(), mChildren.end(), std::back_inserter( lVisibleChildren ),
                      []( auto x ) { return ( x.mItem == nullptr ) || ( x.mItem->mIsVisible ); } );

        float lTaken = mItemSpacing * ( lVisibleChildren.size() - 1 );

        for( auto const &lItem : lVisibleChildren )
        {
            if( ( lItem.mItem != nullptr ) && ( !lItem.mItem->mIsVisible ) )
                continue;

            if( lItem.mIsSeparator )
            {
                lTaken += 1.0f;
            }
            else if( lItem.mFixedSize > 0.0f )
            {
                lTaken += lItem.mFixedSize;
            }
            else if( !lItem.mExpand )
            {
                auto lRequiredSize = lItem.mItem->RequiredSize();

                lTaken += ( mOrientation == eBoxLayoutOrientation::HORIZONTAL ) ? lRequiredSize.x : lRequiredSize.y;
            }
            else
            {
                lExpandCount += 1;
            }
        }

        ImVec2 lExpandedSize{};

        if( mOrientation == eBoxLayoutOrientation::HORIZONTAL )
            lExpandedSize = ImVec2{ ( aSize.x - lTaken ) / lExpandCount, aSize.y };
        else
            lExpandedSize = ImVec2{ aSize.x, ( aSize.y - lTaken ) / lExpandCount };

        ImVec2 lCurrentPosition = aPosition;
        for( auto const &lItem : lVisibleChildren )
        {
            ImVec2 lItemSize{};
            ImVec2 lItemPosition{};
            float  lPositionStep = 0.0f;

            if( ( lItem.mItem != nullptr ) && ( !lItem.mItem->mIsVisible ) )
                continue;

            if( lItem.mExpand && !( lItem.mFixedSize > 0.0f ) )
            {
                lItemSize     = lItem.mFill ? lExpandedSize : ( lItem.mItem ? lItem.mItem->RequiredSize() : ImVec2{} );
                lItemPosition = lItem.mFill ? lCurrentPosition
                                            : GetContentAlignedposition( lItem.mHalign, lItem.mValign, lCurrentPosition, lItemSize,
                                                                         lExpandedSize );
                lPositionStep = ( mOrientation == eBoxLayoutOrientation::VERTICAL ) ? lExpandedSize.y : lExpandedSize.x;
            }
            else if( lItem.mIsSeparator )
            {
                lItemSize.x   = ( mOrientation == eBoxLayoutOrientation::VERTICAL ) ? aSize.x : 1.0;
                lItemSize.y   = ( mOrientation == eBoxLayoutOrientation::VERTICAL ) ? 1.0 : aSize.y;
                lItemPosition = lCurrentPosition - aPosition;
                lPositionStep = 1.0f;
            }
            else if( lItem.mFixedSize > 0.0f )
            {
                if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                    lItemSize = ImVec2{ aSize.x, lItem.mFixedSize };
                else
                    lItemSize = ImVec2{ lItem.mFixedSize, aSize.y };

                lItemSize     = lItem.mFill ? lItemSize : ( lItem.mItem ? lItem.mItem->RequiredSize() : ImVec2{} );
                lItemPosition = lItem.mFill
                                    ? lCurrentPosition
                                    : GetContentAlignedposition( lItem.mHalign, lItem.mValign, lCurrentPosition,
                                                                 ( lItem.mItem ? lItem.mItem->RequiredSize() : ImVec2{} ), lItemSize );
                lPositionStep = lItem.mFixedSize;
            }
            else
            {
                lItemSize = lItem.mItem->RequiredSize();

                if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                    lItemSize.x = aSize.x;
                else
                    lItemSize.y = aSize.y;

                lItemSize     = lItem.mFill ? lItemSize : ( lItem.mItem ? lItem.mItem->RequiredSize() : ImVec2{} );
                lItemPosition = lItem.mFill
                                    ? lCurrentPosition
                                    : GetContentAlignedposition( lItem.mHalign, lItem.mValign, lCurrentPosition,
                                                                 ( lItem.mItem ? lItem.mItem->RequiredSize() : ImVec2{} ), lItemSize );
                lPositionStep = ( mOrientation == eBoxLayoutOrientation::VERTICAL ) ? lItemSize.y : lItemSize.x;
            }

            if( lItem.mItem )
            {
                if( !mSimple )
                {
                    ImGui::SetCursorPos( lItemPosition );
                    ImGui::PushID( (void *)lItem.mItem );
                    ImGui::BeginChild( "##BoxLayoutItem", lItemSize );
                    lItemPosition = ImVec2{};
                }

                lItem.mItem->Update( lItemPosition, lItemSize );
                if( !mSimple )
                {
                    ImGui::EndChild();
                    ImGui::PopID();
                }
            }
            else if( lItem.mIsSeparator )
            {
                ImGuiContext &g      = *GImGui;
                ImGuiWindow  *window = g.CurrentWindow;
                const ImU32   col    = ImGui::GetColorU32( ImGuiCol_Separator );

                if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                {
                    float separator_min = lGlobalPosition.x + lItemPosition.x;
                    float separator_max = separator_min + aSize.x;

                    window->DrawList->AddLine( ImVec2( separator_min, lGlobalPosition.y + lItemPosition.y ),
                                               ImVec2( separator_max, lGlobalPosition.y + lItemPosition.y ), col, 1.0f );
                }
                else
                {
                    float separator_min = lGlobalPosition.y + lItemPosition.y;
                    float separator_max = separator_min + aSize.y;

                    window->DrawList->AddLine( ImVec2( lGlobalPosition.x + lItemPosition.x, separator_min ),
                                               ImVec2( lGlobalPosition.x + lItemPosition.x, separator_max ), col, 1.0f );
                }
            }

            if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                lCurrentPosition.y += ( lPositionStep + mItemSpacing );
            else
                lCurrentPosition.x += ( lPositionStep + mItemSpacing );
        }
    }
} // namespace SE::Core