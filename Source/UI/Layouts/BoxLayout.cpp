#include "BoxLayout.h"

namespace SE::Core
{

    BoxLayout::BoxLayout( eBoxLayoutOrientation aOrientation )
        : mOrientation{ aOrientation }
    {
    }

    void BoxLayout::PushStyles() {}
    void BoxLayout::PopStyles() {}

    ImVec2 BoxLayout::RequiredSize() { return ImVec2{}; }

    BoxLayout &BoxLayout::SetItemSpacing( float aItemSpacing )
    {
        mItemSpacing = aItemSpacing;

        return *this;
    }

    BoxLayout &BoxLayout::Add( UIComponent *aChild, bool aExpand, bool aFill )
    {
        return Add( aChild, 0.0f, aExpand, aFill, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER );
    }

    BoxLayout &BoxLayout::Add( UIComponent *aChild, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                               eVerticalAlignment const &aVAlignment )
    {
        return Add( aChild, 0.0f, aExpand, aFill, aHAlignment, aVAlignment );
    }

    BoxLayout &BoxLayout::Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill )
    {
        return Add( aChild, 0.0f, aExpand, aFill, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER );
    }

    BoxLayout &BoxLayout::Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill,
                               eHorizontalAlignment const &aHAlignment, eVerticalAlignment const &aVAlignment )
    {
        mChildren.push_back( BoxLayoutItem{ aChild, 0.0f, aExpand, aFill, aHAlignment, aVAlignment } );

        return *this;
    }

    void BoxLayout::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        uint32_t lExpandCount = 0;
        float    lTaken       = mItemSpacing * ( mChildren.size() - 1 );

        for( auto const &lItem : mChildren )
        {
            lExpandCount += lItem.mExpand ? 1 : 0;

            if( lItem.mFixedSize > 0.0f )
            {
                lTaken += lItem.mFixedSize;
            }
            else if( !lItem.mExpand )
            {
                auto lRequiredSize = lItem.mItem->RequiredSize();

                lTaken += ( mOrientation == eBoxLayoutOrientation::HORIZONTAL ) ? lRequiredSize.x : lRequiredSize.y;
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
            if( lItem.mExpand )
            {
                lItemSize     = lItem.mFill ? lExpandedSize : lItem.mItem->RequiredSize();
                lItemPosition = lItem.mFill ? lCurrentPosition
                                            : GetContentAlignedposition( lItem.mHalign, lItem.mValign, lCurrentPosition, lItemSize,
                                                                         lExpandedSize );
                lPositionStep = ( mOrientation == eBoxLayoutOrientation::VERTICAL ) ? lExpandedSize.y : lExpandedSize.x;
            }
            else if( lItem.mFixedSize > 0.0f )
            {
                if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                    lItemSize = ImVec2{ lItem.mFixedSize, aSize.y };
                else
                    lItemSize = ImVec2{ aSize.x, lItem.mFixedSize };

                lItemSize     = lItem.mFill ? lItemSize : lItem.mItem->RequiredSize();
                lItemPosition = lItem.mFill ? lCurrentPosition
                                            : GetContentAlignedposition( lItem.mHalign, lItem.mValign, lCurrentPosition,
                                                                         lItem.mItem->RequiredSize(), lItemSize );
                lPositionStep = lItem.mFixedSize;
            }
            else
            {
                lItemSize = lItem.mItem->RequiredSize();

                if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                    lItemSize.x = aSize.x;
                else
                    lItemSize.y = aSize.y;

                lItemSize     = lItem.mFill ? lItemSize : lItem.mItem->RequiredSize();
                lItemPosition = lItem.mFill ? lCurrentPosition
                                            : GetContentAlignedposition( lItem.mHalign, lItem.mValign, lCurrentPosition,
                                                                         lItem.mItem->RequiredSize(), lItemSize );
                lPositionStep = ( mOrientation == eBoxLayoutOrientation::VERTICAL ) ? lItemSize.y : lItemSize.x;
            }

            if( lItem.mItem ) lItem.mItem->Update( lItemPosition, lItemSize );

            if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                lCurrentPosition.y += ( lPositionStep + mItemSpacing );
            else
                lCurrentPosition.x += ( lPositionStep + mItemSpacing );
        }
    }
} // namespace SE::Core