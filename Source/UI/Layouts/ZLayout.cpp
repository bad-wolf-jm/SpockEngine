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
            if( ( lItem.mSize.x > 0.0f ) && ( lItem.mSize.y > 0.0f ) )
            {
                lWidth  = math::max( lWidth, lItem.mSize.x );
                lHeight = math::max( lHeight, lItem.mSize.y );
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
        Add( aChild, math::vec2{-1.0f, -1.0f}, math::vec2{-1.0f, -1.0f}, aExpand, aFill, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER );
    }

    void UIZLayout::Add( UIComponent *aChild, bool aExpand, bool aFill, eHorizontalAlignment const &aHAlignment,
                         eVerticalAlignment const &aVAlignment )
    {
        Add( aChild, math::vec2{-1.0f, -1.0f}, math::vec2{-1.0f, -1.0f}, aExpand, aFill, aHAlignment, aVAlignment );
    }

    void UIZLayout::Add( UIComponent *aChild, math::vec2 aSize, math::vec2 aPosition, bool aExpand, bool aFill )
    {
        Add( aChild, aSize, aPosition, aExpand, aFill, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER );
    }

    void UIZLayout::Add( UIComponent *aChild, math::vec2 aSize, math::vec2 aPosition, bool aExpand, bool aFill,
                         eHorizontalAlignment const &aHAlignment, eVerticalAlignment const &aVAlignment )
    {
        mChildren.push_back( ZLayoutItem{ aChild, ImVec2{ aSize.x, aSize.y }, ImVec2{ aPosition.x, aPosition.y }, aExpand, aFill,
                                          aHAlignment, aVAlignment } );
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
            }
            else if( ( lItem.mSize.x > 1.0f ) && ( lItem.mSize.y > 1.0f ) )
            {
                lItemSize = lItem.mFill ? lItem.mSize : lItem.mItem->RequiredSize();
            }
            else if( ( lItem.mSize.x >= 0.0f ) && ( lItem.mSize.y >= 0.0f ) && ( lItem.mSize.x <= 1.0f ) && ( lItem.mSize.y <= 1.0f ) )
            {
                lItemSize = lItem.mSize * aSize;
            }
            else
            {
                lItemSize = lItem.mFill ? aSize : lItem.mItem->RequiredSize();
            }

            if( lItem.mExpand )
            {
                lItemPosition =
                    lItem.mFill ? aPosition : GetContentAlignedposition( lItem.mHalign, lItem.mValign, aPosition, lItemSize, aSize );
            }
            else if( ( lItem.mPosition.x > 1.0f ) && ( lItem.mPosition.y > 1.0f ) )
            {
                lItemPosition =
                    lItem.mFill ? aPosition : GetContentAlignedposition( lItem.mHalign, lItem.mValign, aPosition, lItemSize, aSize );
            }
            else if( ( lItem.mPosition.x >= 0.0f ) && ( lItem.mPosition.y >= 0.0f ) && ( lItem.mPosition.x <= 1.0f ) &&
                     ( lItem.mPosition.y <= 1.0f ) )
            {
                lItemPosition = aPosition + lItem.mPosition * aSize;
            }
            else
            {
                lItemPosition =
                    lItem.mFill ? aPosition : GetContentAlignedposition( lItem.mHalign, lItem.mValign, aPosition, lItemSize, aSize );
            }

            if( lItem.mItem ) lItem.mItem->Update( lItemPosition, lItemSize );
        }
    }
} // namespace SE::Core