#include "BoxLayout.h"

namespace SE::Core
{

    UIBoxLayout::UIBoxLayout( eBoxLayoutOrientation aOrientation )
        : mOrientation{ aOrientation }
    {
    }

    void UIBoxLayout::PushStyles() {}
    void UIBoxLayout::PopStyles() {}

    ImVec2 UIBoxLayout::RequiredSize()
    {
        float lWidth  = 0.0f;
        float lHeight = 0.0f;

        for( auto const &lItem : mChildren )
        {
            if( ( lItem.mItem != nullptr ) && ( !lItem.mItem->mIsVisible ) ) continue;
            
            ImVec2 lRequiredSize{};
            if( lItem.mFixedSize > 0.0f )
            {
                if( lItem.mItem ) lRequiredSize = lItem.mItem->RequiredSize();

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

                if( lItem.mItem ) lRequiredSize = lItem.mItem->RequiredSize();

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

        return ImVec2{ lWidth, lHeight };
    }

    void UIBoxLayout::SetItemSpacing( float aItemSpacing ) { mItemSpacing = aItemSpacing; }

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
        mChildren.push_back( BoxLayoutItem{ aChild, aFixedSize, aExpand, aFill, aHAlignment, aVAlignment } );
    }

    void UIBoxLayout::Clear() { mChildren.clear(); }

    void UIBoxLayout::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        uint32_t lExpandCount = 0;

        std::vector<BoxLayoutItem> lVisibleChildren;
        std::copy_if( mChildren.begin(), mChildren.end(), std::back_inserter( lVisibleChildren ),
                      []( auto x ) { return (x.mItem != nullptr) && (x.mItem->mIsVisible); } );

        float lTaken = mItemSpacing * ( lVisibleChildren.size() - 1 );

        for( auto const &lItem : lVisibleChildren )
        {
            if( ( lItem.mItem != nullptr ) && ( !lItem.mItem->mIsVisible ) ) continue;

            if( lItem.mFixedSize > 0.0f )
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
        for( auto const &lItem : mChildren )
        {
            ImVec2 lItemSize{};
            ImVec2 lItemPosition{};
            float  lPositionStep = 0.0f;

            if( ( lItem.mItem != nullptr ) && ( !lItem.mItem->mIsVisible ) ) continue;

            if( lItem.mExpand && !( lItem.mFixedSize > 0.0f ) )
            {
                lItemSize     = lItem.mFill ? lExpandedSize : ( lItem.mItem ? lItem.mItem->RequiredSize() : ImVec2{} );
                lItemPosition = lItem.mFill ? lCurrentPosition
                                            : GetContentAlignedposition( lItem.mHalign, lItem.mValign, lCurrentPosition, lItemSize,
                                                                         lExpandedSize );
                lPositionStep = ( mOrientation == eBoxLayoutOrientation::VERTICAL ) ? lExpandedSize.y : lExpandedSize.x;
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
                ImGui::SetCursorPos( lItemPosition );
                ImGui::PushID( (void *)lItem.mItem );
                ImGui::BeginChild( "##BoxLayoutItem", lItemSize );
                lItem.mItem->Update( ImVec2{}, lItemSize );
                ImGui::EndChild();
                ImGui::PopID();
            }

            if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                lCurrentPosition.y += ( lPositionStep + mItemSpacing );
            else
                lCurrentPosition.x += ( lPositionStep + mItemSpacing );
        }
    }

    void *UIBoxLayout::UIBoxLayout_CreateWithOrientation( eBoxLayoutOrientation aOrientation )
    {
        auto lNewLayout = new UIBoxLayout( aOrientation );

        return static_cast<void *>( lNewLayout );
    }

    void UIBoxLayout::UIBoxLayout_Destroy( void *aInstance ) { delete static_cast<UIBoxLayout *>( aInstance ); }

    void UIBoxLayout::UIBoxLayout_AddAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill,
                                                      eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIBoxLayout::UIBoxLayout_AddNonAlignedNonFixed( void *aInstance, void *aChild, bool aExpand, bool aFill )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aExpand, aFill );
    }

    void UIBoxLayout::UIBoxLayout_AddAlignedFixed( void *aInstance, void *aChild, float aFixedSize, bool aExpand, bool aFill,
                                                   eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aFixedSize, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIBoxLayout::UIBoxLayout_AddNonAlignedFixed( void *aInstance, void *aChild, float aFixedSize, bool aExpand, bool aFill )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aFixedSize, aExpand, aFill );
    }

    void UIBoxLayout::UIBoxLayout_SetItemSpacing( void *aInstance, float aItemSpacing )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );

        lInstance->SetItemSpacing( aItemSpacing );
    }

    void UIBoxLayout::UIBoxLayout_Clear( void *aInstance )
    {
        auto lInstance = static_cast<UIBoxLayout *>( aInstance );

        lInstance->Clear();
    }

} // namespace SE::Core