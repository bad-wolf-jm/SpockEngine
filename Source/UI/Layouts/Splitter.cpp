#include "Splitter.h"

#include "imgui_internal.h"

namespace SE::Core
{

    UISplitter::UISplitter( eBoxLayoutOrientation aOrientation )
        : mOrientation{ aOrientation }
    {
    }

    void UISplitter::PushStyles()
    {
    }
    void UISplitter::PopStyles()
    {
    }

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

    void UISplitter::SetOrientation( eBoxLayoutOrientation aValue )
    {
        mOrientation = aValue;
    }

    void UISplitter::SetItemSpacing( float aItemSpacing )
    {
        mItemSpacing = aItemSpacing;
    }

    void UISplitter::Add1( UIComponent *aChild )
    {
        mChild1 = aChild;
    }

    void UISplitter::Add2( UIComponent *aChild )
    {
        mChild2 = aChild;
    }

    static bool Splitter( eBoxLayoutOrientation aOrientation, float aThickness, float *aSize1, float *aSize2, float aMinSize1,
                          float aMinSize2, float aLength = -1.0f )
    {
        using namespace ImGui;
        ImGuiContext &g      = *GImGui;
        ImGuiWindow  *window = g.CurrentWindow;
        ImGuiID       id     = window->GetID( "##Splitter" );
        ImRect        bb;

        if( aOrientation == eBoxLayoutOrientation::VERTICAL )
        {
            bb.Min = window->DC.CursorPos + ImVec2( *aSize1, 0.0f );
            bb.Max = bb.Min + CalcItemSize( ImVec2( aThickness, aLength ), 0.0f, 0.0f );
        }
        else
        {
            bb.Min = window->DC.CursorPos + ImVec2( 0.0f, *aSize1 );
            bb.Max = bb.Min + CalcItemSize( ImVec2( aLength, aThickness ), 0.0f, 0.0f );
        }

        ImGuiAxis lSplitDirection = ( aOrientation == eBoxLayoutOrientation::VERTICAL ) ? ImGuiAxis_X : ImGuiAxis_Y;
        return SplitterBehavior( bb, id, lSplitDirection, aSize1, aSize2, aMinSize1, aMinSize2, 0.0f );
    }

    void UISplitter::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        auto lSize     = aSize - GetContentPadding();
        auto lPosition = aPosition + GetContentOffset();

        if( mSizeSet && ( ( lSize.x != mCurrentSize.x ) || ( lSize.y != mCurrentSize.y ) ) )
        {
            if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                mSize2 = lSize.x - mSize1 - mItemSpacing;
            else
                mSize2 = lSize.y - mSize1 - mItemSpacing;

            mCurrentSize = lSize;
        }

        if( !mSizeSet )
        {
            if( mOrientation == eBoxLayoutOrientation::VERTICAL )
                mSize1 = mSize2 = ( lSize.x - mItemSpacing ) * 0.5f;
            else
                mSize1 = mSize2 = ( lSize.y - mItemSpacing ) * 0.5f;

            mSizeSet = true;
        }

        ImGui::SetCursorPos( lPosition );
        Splitter( mOrientation, mItemSpacing, &mSize1, &mSize2, 50.0f, 50.0f,
                  ( mOrientation == eBoxLayoutOrientation::VERTICAL ) ? lSize.y : lSize.x );

        ImVec2 lTopLeft = ImGui::GetCursorScreenPos();
        if( mChild1 )
        {
            ImVec2 lItemSize =
                ( mOrientation == eBoxLayoutOrientation::HORIZONTAL ) ? ImVec2{ lSize.x, mSize1 } : ImVec2{ mSize1, lSize.y };

            ImGui::SetCursorPos( lPosition );
            ImGui::PushID( (void *)mChild1 );
            ImGui::BeginChild( "##Child_1", lItemSize );
            mChild1->Update( ImVec2{}, lItemSize );
            ImGui::EndChild();
            ImGui::PopID();
        }

        ImVec2 lSecondItemPosition{};
        if( mOrientation == eBoxLayoutOrientation::VERTICAL )
            lSecondItemPosition.x = mSize1 + mItemSpacing;
        else
            lSecondItemPosition.y = mSize1 + mItemSpacing;

        if( mChild2 )
        {
            ImVec2 lItemSize =
                ( mOrientation == eBoxLayoutOrientation::HORIZONTAL ) ? ImVec2{ lSize.x, mSize2 } : ImVec2{ mSize2, lSize.y };
            ImGui::SetCursorPos( lPosition + lSecondItemPosition );
            ImGui::PushID( (void *)mChild2 );
            ImGui::BeginChild( "##Child_2", lItemSize );
            mChild2->Update( ImVec2{}, lItemSize );
            ImGui::EndChild();
            ImGui::PopID();
        }
    }
} // namespace SE::Core