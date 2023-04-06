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
        ImVec2 lLeftSize  = mChild1 ? mChild1->RequiredSize() : ImVec2{};
        ImVec2 lRightSize = mChild2 ? mChild2->RequiredSize() : ImVec2{};

        float lWidth  = 0.0f;
        float lHeight = 0.0f;

        if( mOrientation == eBoxLayoutOrientation::HORIZONTAL )
        {
            lWidth  = math::max( lLeftSize.x, lRightSize.x );
            lHeight = lLeftSize.y + mItemSpacing + lRightSize.y;
        }
        else
        {
            lWidth  = lLeftSize.x + mItemSpacing + lRightSize.x;
            lHeight = math::max( lLeftSize.y, lRightSize.y );
        }

        return ImVec2{ lWidth, lHeight };
    }

    void UISplitter::SetItemSpacing( float aItemSpacing ) { mItemSpacing = aItemSpacing; }

    void UISplitter::Add1( UIComponent *aChild ) { mChild1 = aChild; }

    void UISplitter::Add2( UIComponent *aChild ) { mChild2 = aChild; }

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
        if( !mSizeSet )
        {
            if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                mSize1 = mSize2 = ( aSize.x - mItemSpacing ) * 0.5f;
            else
                mSize1 = mSize2 = ( aSize.y - mItemSpacing ) * 0.5f;
            mSizeSet = true;
        }
        Splitter( mOrientation, mItemSpacing, &mSize1, &mSize2, 50.0f, 50.0f );

        if( mChild1 )
        {
            ImVec2 lItemSize =
                ( mOrientation == eBoxLayoutOrientation::HORIZONTAL ) ? ImVec2{ aSize.x, mSize1 } : ImVec2{ mSize1, aSize.y };
            mChild1->Update( aPosition, lItemSize );
        }

        ImVec2 lSecondItemPosition{};
        if( mOrientation == eBoxLayoutOrientation::VERTICAL )
            lSecondItemPosition.x = mSize1 + mItemSpacing;
        else
            lSecondItemPosition.y = mSize1 + mItemSpacing;

        if( mChild2 )
        {
            ImVec2 lItemSize =
                ( mOrientation == eBoxLayoutOrientation::HORIZONTAL ) ? ImVec2{ aSize.x, mSize2 } : ImVec2{ mSize2, aSize.y };
            mChild2->Update( aPosition + lSecondItemPosition, lItemSize );
        }
    }

    void *UISplitter::UISplitter_Create()
    {
        auto lNewLayout = new UISplitter();

        return static_cast<void *>( lNewLayout );
    }

    void *UISplitter::UISplitter_CreateWithOrientation( eBoxLayoutOrientation aOrientation )
    {
        auto lNewLayout = new UISplitter( aOrientation );

        return static_cast<void *>( lNewLayout );
    }

    void UISplitter::UISplitter_Destroy( void *aInstance ) { delete static_cast<UISplitter *>( aInstance ); }

    void UISplitter::UISplitter_Add1( void *aInstance, void *aChild )
    {
        auto lInstance = static_cast<UISplitter *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add1( lChild );
    }

    void UISplitter::UISplitter_Add2( void *aInstance, void *aChild )
    {
        auto lInstance = static_cast<UISplitter *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );

        lInstance->Add2( lChild );
    }

    void UISplitter::UISplitter_SetItemSpacing( void *aInstance, float aItemSpacing )
    {
        auto lInstance = static_cast<UISplitter *>( aInstance );

        lInstance->SetItemSpacing( aItemSpacing );
    }
} // namespace SE::Core