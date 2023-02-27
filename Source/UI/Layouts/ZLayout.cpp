#include "ZLayout.h"

namespace SE::Core
{
    void UIZLayout::PushStyles() {}
    void UIZLayout::PopStyles() {}

    ImVec2 UIZLayout::RequiredSize()
    {
        float lWidth  = 0.0f;
        float lHeight = 0.0f;

        for( auto const &lItem : mChildren )
        {
            if( ( lItem.mFixedSize.x > 0.0f ) && ( lItem.mFixedSize.y > 0.0f ) )
            {
                lWidth  = math::max( lWidth, lItem.mFixedSize.x );
                lHeight = math::max( lHeight, lItem.mFixedSize.y );
            }
            else
            {
                ImVec2 lRequiredSize{};
                if( lItem.mItem ) lRequiredSize = lItem.mItem->RequiredSize();
                lWidth  = math::max( lWidth, lRequiredSize.x );
                lHeight = math::max( lHeight, lRequiredSize.y );
            }
        }

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
        mChildren.push_back( ZLayoutItem{ aChild, ImVec2{ aFixedSize.x, aFixedSize.y }, aExpand, aFill, aHAlignment, aVAlignment } );
    }

    void UIZLayout::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        for( auto const &lItem : mChildren )
        {
            ImVec2 lItemSize{};
            ImVec2 lItemPosition{};

            if( lItem.mExpand )
            {
                lItemSize = lItem.mFill ? aSize : lItem.mItem->RequiredSize();
                lItemPosition =
                    lItem.mFill ? aPosition : GetContentAlignedposition( lItem.mHalign, lItem.mValign, aPosition, lItemSize, aSize );
            }
            else if( ( lItem.mFixedSize.x > 0.0f ) && ( lItem.mFixedSize.y > 0.0f ) )
            {
                lItemSize = lItem.mFill ? lItem.mFixedSize : lItem.mItem->RequiredSize();
                lItemPosition =
                    lItem.mFill ? aPosition : GetContentAlignedposition( lItem.mHalign, lItem.mValign, aPosition, lItemSize, aSize );
            }
            else
            {
                lItemSize = lItem.mFill ? aSize : lItem.mItem->RequiredSize();
                lItemPosition =
                    lItem.mFill ? aPosition : GetContentAlignedposition( lItem.mHalign, lItem.mValign, aPosition, lItemSize, aSize );
            }

            if( lItem.mItem ) lItem.mItem->Update( lItemPosition, lItemSize );
        }
    }
} // namespace SE::Core