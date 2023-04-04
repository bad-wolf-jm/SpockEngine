#include "Splitter.h"

#include "imgui_internal.h"

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
            // }
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

    static bool Splitter( eBoxLayoutOrientation aOrientation, float aThickness, float *aSize1, float *aSize2, float aMinSize1,
                          float aMinSize2 )
    {
        using namespace ImGui;
        ImGuiContext &g      = *GImGui;
        ImGuiWindow  *window = g.CurrentWindow;
        ImGuiID       id     = window->GetID( "##Splitter" );
        ImRect        bb;

        if( aOrientation == eBoxLayoutOrientation::VERTICAL )
        {
            bb.Min = window->DC.CursorPos + ImVec2( *aSize1, 0.0f );
            bb.Max = bb.Min + CalcItemSize( ImVec2( aThickness, -1.0f ), 0.0f, 0.0f );
        }
        else
        {
            bb.Min = window->DC.CursorPos + ImVec2( 0.0f, *aSize1 );
            bb.Max = bb.Min + CalcItemSize( ImVec2( -1.0f, aThickness ), 0.0f, 0.0f );
        }

        ImGuiAxis lSplitDirection = ( aOrientation == eBoxLayoutOrientation::VERTICAL ) ? ImGuiAxis_X : ImGuiAxis_Y;
        return SplitterBehavior( bb, id, lSplitDirection, aSize1, aSize2, aMinSize1, aMinSize2, 0.0f );
    }

    void UISplitter::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        std::vector<float> lCumulativeSizes( mChildren.size() );

        lCumulativeSizes[mChildren.size() - 1] = mItemSizes[mChildren.size() - 1];

        for( uint32_t i = mChildren.size() - 2; i >= 0; i-- ) lCumulativeSizes[i] = lCumulativeSizes[i + 1] + mItemSizes[i];

        ImVec2 lCurrentPosition = aPosition;
        for( uint32_t i = 0; i < mChildren.size(); i++ )
        {
            auto  &lItem = mChildren[i];
            ImVec2 lItemPosition{};
            float  lPositionStep = 0.0f;

            ImVec2 lItemSize{};
            if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                lItemSize = ImVec2{ aSize.x, mItemSizes[i] };
            else
                lItemSize = ImVec2{ mItemSizes[i], aSize.y };

            lItemSize     = lItem.mFill ? lItemSize : ( lItem.mItem ? lItem.mItem->RequiredSize() : ImVec2{} );
            lItemPosition = lItem.mFill
                                ? lCurrentPosition
                                : GetContentAlignedposition( lItem.mHalign, lItem.mValign, lCurrentPosition,
                                                             ( lItem.mItem ? lItem.mItem->RequiredSize() : ImVec2{} ), lItemSize );
            lPositionStep = ( mOrientation == eBoxLayoutOrientation::VERTICAL ) ? lItemSize.y : lItemSize.x;

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