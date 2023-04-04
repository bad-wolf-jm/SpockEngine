#include "Splitter.h"

namespace SE::Core
{

    UISplitter::UISplitter( eBoxLayoutOrientation aOrientation )
        : mOrientation{ aOrientation }
    {
    }

    void UISplitter::PushStyles() {}
    void UISplitter::PopStyles() {}

    ImVec2 UISplitter::RequiredSize()
    {
        float lWidth  = 0.0f;
        float lHeight = 0.0f;

        for( auto const &lItem : mChildren )
        {
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

    void UISplitter::SetItemSpacing( float aItemSpacing ) { mItemSpacing = aItemSpacing; }

    void UISplitter::Add( UIComponent *aChild )
    {
        mChildren.push_back( SplitterItem{ aChild, true, eHorizontalAlignment::CENTER, eVerticalAlignment::CENTER } );
    }

    void UISplitter::Add( UIComponent *aChild, eHorizontalAlignment const &aHAlignment, eVerticalAlignment const &aVAlignment )
    {
        mChildren.push_back( SplitterItem{ aChild, false, aHAlignment, aVAlignment } );
    }

    void UISplitter::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        uint32_t lExpandCount = 0;
        float    lTaken       = mItemSpacing * ( mChildren.size() - 1 );

        for( auto const &lItem : mChildren )
        {
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

            if( lItem.mItem ) lItem.mItem->Update( lItemPosition, lItemSize );

            if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                lCurrentPosition.y += ( lPositionStep + mItemSpacing );
            else
                lCurrentPosition.x += ( lPositionStep + mItemSpacing );
        }
    }

    void *UISplitter::UISplitter_CreateWithOrientation( eBoxLayoutOrientation aOrientation )
    {
        auto lNewLayout = new UISplitter( aOrientation );

        return static_cast<void *>( lNewLayout );
    }

    void UISplitter::UISplitter_Destroy( void *aInstance ) { delete static_cast<UISplitter *>( aInstance ); }

    void UISplitter::UISplitter_AddFill( void *aInstance, void *aChild )
    {
        auto lInstance = static_cast<UISplitter *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild );
    }

    void UISplitter::UISplitter_AddAligned( void *aInstance, void *aChild, eHorizontalAlignment aHAlignment,
                                            eVerticalAlignment aVAlignment )
    {
        auto lInstance = static_cast<UISplitter *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add( lChild, aHAlignment, aVAlignment );
    }

    void UISplitter::UISplitter_SetItemSpacing( void *aInstance, float aItemSpacing )
    {
        auto lInstance = static_cast<UISplitter *>( aInstance );

        lInstance->SetItemSpacing( aItemSpacing );
    }
} // namespace SE::Core