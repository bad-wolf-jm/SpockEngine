#include "ZLayout.h"

namespace SE::Core
{
    void UIZLayout::PushStyles() {}
    void UIZLayout::PopStyles() {}

    ImVec2 UIZLayout::RequiredSize()
    {
        float lWidth  = 0.0f;
        float lHeight = 0.0f;

        // for( auto const &lItem : mChildren )
        // {
        //     ImVec2 lRequiredSize{};
        //     if( lItem.mFixedSize > 0.0f )
        //     {
        //         if( lItem.mItem ) lRequiredSize = lItem.mItem->RequiredSize();

        //         if( mOrientation == eZLayoutOrientation::HORIZONTAL )
        //         {
        //             lWidth += lItem.mFixedSize;
        //             lHeight = math::max( lHeight, lRequiredSize.y );
        //         }
        //         else
        //         {
        //             lHeight += lItem.mFixedSize;
        //             lWidth = math::max( lWidth, lRequiredSize.x );
        //         }
        //     }
        //     else
        //     {

        //         if( lItem.mItem ) lRequiredSize = lItem.mItem->RequiredSize();

        //         if( mOrientation == eZLayoutOrientation::HORIZONTAL )
        //         {
        //             lWidth += lRequiredSize.x;
        //             lHeight = math::max( lHeight, lRequiredSize.y );
        //         }
        //         else
        //         {
        //             lHeight += lRequiredSize.y;
        //             lWidth = math::max( lWidth, lRequiredSize.x );
        //         }
        //     }
        // }

        return ImVec2{ lWidth, lHeight };
    }

    void UIZLayout::Add( UIComponent *aChild, bool aExpand, bool aFill )
    {
        Add( aChild, math::vec2{}, aExpand, aFill, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER );
    }

    void UIZLayout::Add( UIComponent *aChild, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                         eVerticalAlignment const &aVAlignment )
    {
        Add( aChild, math::vec2{}, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIZLayout::Add( UIComponent *aChild, math::vec2 aFixedSize, bool aExpand, bool aFill )
    {
        Add( aChild, aFixedSize, aExpand, aFill, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER );
    }

    void UIZLayout::Add( UIComponent *aChild, math::vec2 aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                         eVerticalAlignment const &aVAlignment )
    {
        mChildren.push_back( ZLayoutItem{ aChild, aFixedSize, aExpand, aFill, aHAlignment, aVAlignment } );
    }

    void UIZLayout::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        // uint32_t lExpandCount = 0;
        // float    lTaken       = mItemSpacing * ( mChildren.size() - 1 );

        // for( auto const &lItem : mChildren )
        // {
        //     lExpandCount += lItem.mExpand ? 1 : 0;

        //     if( lItem.mFixedSize > 0.0f )
        //     {
        //         lTaken += lItem.mFixedSize;
        //     }
        //     else if( !lItem.mExpand )
        //     {
        //         auto lRequiredSize = lItem.mItem->RequiredSize();

        //         lTaken += ( mOrientation == eZLayoutOrientation::HORIZONTAL ) ? lRequiredSize.x : lRequiredSize.y;
        //     }
        // }

        // ImVec2 lExpandedSize{};

        // if( mOrientation == eZLayoutOrientation::HORIZONTAL )
        //     lExpandedSize = ImVec2{ ( aSize.x - lTaken ) / lExpandCount, aSize.y };
        // else
        //     lExpandedSize = ImVec2{ aSize.x, ( aSize.y - lTaken ) / lExpandCount };

        // ImVec2 lCurrentPosition = aPosition;
        // for( auto const &lItem : mChildren )
        // {
        //     ImVec2 lItemSize{};
        //     ImVec2 lItemPosition{};
        //     float  lPositionStep = 0.0f;

        //     if( lItem.mExpand )
        //     {
        //         lItemSize     = lItem.mFill ? lExpandedSize : lItem.mItem->RequiredSize();
        //         lItemPosition = lItem.mFill ? lCurrentPosition
        //                                     : GetContentAlignedposition( lItem.mHalign, lItem.mValign, lCurrentPosition, lItemSize,
        //                                                                  lExpandedSize );
        //         lPositionStep = ( mOrientation == eZLayoutOrientation::VERTICAL ) ? lExpandedSize.y : lExpandedSize.x;
        //     }
        //     else if( lItem.mFixedSize > 0.0f )
        //     {
        //         if( mOrientation == eZLayoutOrientation::VERTICAL )
        //             lItemSize = ImVec2{ lItem.mFixedSize, aSize.y };
        //         else
        //             lItemSize = ImVec2{ aSize.x, lItem.mFixedSize };

        //         lItemSize     = lItem.mFill ? lItemSize : lItem.mItem->RequiredSize();
        //         lItemPosition = lItem.mFill ? lCurrentPosition
        //                                     : GetContentAlignedposition( lItem.mHalign, lItem.mValign, lCurrentPosition,
        //                                                                  lItem.mItem->RequiredSize(), lItemSize );
        //         lPositionStep = lItem.mFixedSize;
        //     }
        //     else
        //     {
        //         lItemSize = lItem.mItem->RequiredSize();

        //         if( mOrientation == eZLayoutOrientation::VERTICAL )
        //             lItemSize.x = aSize.x;
        //         else
        //             lItemSize.y = aSize.y;

        //         lItemSize     = lItem.mFill ? lItemSize : lItem.mItem->RequiredSize();
        //         lItemPosition = lItem.mFill ? lCurrentPosition
        //                                     : GetContentAlignedposition( lItem.mHalign, lItem.mValign, lCurrentPosition,
        //                                                                  lItem.mItem->RequiredSize(), lItemSize );
        //         lPositionStep = ( mOrientation == eZLayoutOrientation::VERTICAL ) ? lItemSize.y : lItemSize.x;
        //     }

        //     if( lItem.mItem ) lItem.mItem->Update( lItemPosition, lItemSize );

        //     if( mOrientation == eZLayoutOrientation::VERTICAL )
        //         lCurrentPosition.y += ( lPositionStep + mItemSpacing );
        //     else
        //         lCurrentPosition.x += ( lPositionStep + mItemSpacing );
        // }
    }
} // namespace SE::Core