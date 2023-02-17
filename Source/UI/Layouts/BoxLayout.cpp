#include "BoxLayout.h"

namespace SE::Core
{
    BoxLayout &BoxLayout::Add( UIComponent *aChild, bool aExpand, bool aFill ) { return Add( aChild, 0.0f, aExpand, aFill ); }

    BoxLayout &BoxLayout::Add( UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill )
    {
        mChildren.emplace_back( { aChild, 0.0f, aExpand, aFill } );

        return *this;
    }

    void BoxLayout::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        uint32_t lExpandCount = 0;
        float    lTaken       = 0;

        for( auto const &lItem : mChildren )
        {
            lExpandCount += lItem.mExpand ? 1 : 0;
            if( lItem.mFixedSize > 0.0f )
            {
                lTaken += mFixedSize;
            }
            else
            {
                auto lRequiredSize = lItem.mItem->RequiredSize();

                lTaken += ( mOrientation == eBoxLayoutOrientation::HORIZONTAL ) ? lRequiredSize.x : lRequiredSize.y;
            }
        }

        ImVec2 lExpandedSize{};

        if( mOrientation == eBoxLayoutOrientation::HORIZONTAL )
        {
            lExpandedSize = ImVec2{ aSize.x / lExpandCount, aSize.y };
        }
        else
        {
            lExpandedSize = ImVec2{ aSize.x, aSize.y / lExpandCount };
        }

        ImVec2 lCurrentPosition = aPosition;
        for( auto const &lItem : mChildren )
        {
            ImVec2 lItemSize{};
            if( lItem.mExpand )
            {
                lItemSize = lItem.mFill ? lExpandedSize : lItem.mItem->RequiredSize();

                if( lItem.mItem )
                {
                    lItem.mItem->Update( lCurrentPosition, lItemSize );
                }

                if( mOrientation == eBoxLayoutOrientation::HORIZONTAL )
                    lCurrentPosition.y += lExpandedSize.y;
                else
                    lCurrentPosition.x += lExpandedSize.x;
            }
            else if( lItem.mFixedSize > 0.0f )
            {
                if( mOrientation == eBoxLayoutOrientation::HORIZONTAL )
                    lItemSize = ImVec2{ lItem.mFixedSize, aSize.y };
                else
                    lItemSize = ImVec2{ aSize.x, lItem.mFixedSize };

                lItemSize = lItem.mFill ? lItemSize : lItem.mItem->RequiredSize();

                if( lItem.mItem )
                {
                    lItem.mItem->Update( lCurrentPosition, lItemSize );
                }

                if( mOrientation == eBoxLayoutOrientation::HORIZONTAL )
                    lCurrentPosition.y += lItem.mFixedSize;
                else
                    lCurrentPosition.x += lItem.mFixedSize;
            }
            else
            {
                lItemSize = lItem.mItem->RequiredSize();

                if( lItem.mItem )
                {
                    lItem.mItem->Update( lCurrentPosition, lItemSize );
                }

                if( mOrientation == eBoxLayoutOrientation::HORIZONTAL )
                    lCurrentPosition.y += lItemSize.y;
                else
                    lCurrentPosition.x += lItemSize.x;
            }
        }
    }
} // namespace SE::Core