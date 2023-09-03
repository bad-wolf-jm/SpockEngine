#include "TreeView.h"

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

namespace SE::Core
{
    UITreeViewNode::UITreeViewNode( UITreeView *aTreeView, UITreeViewNode *aParent )
        : mTreeView{ aTreeView }
        , mParent{ aParent }
        , mFlags{ ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_FramePadding }
    {
        mImage = New<UIStackLayout>();
        mImage->SetPadding( 0.0f );

        mText = New<UILabel>( "" );
        mText->SetPadding( 0.0f );
        mText->mIsVisible = true;

        mIndicator = New<UIStackLayout>();
        mIndicator->SetPadding( 0.0f );

        mLayout = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mLayout->SetSimple( true );
        mLayout->SetPadding( 0.0f );

        mText->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mLayout->Add( mText.get(), true, true );
        mLayout->Add( mIndicator.get(), 20.0f, false, true );

        mLayout->mIsVisible    = true;
        mImage->mIsVisible     = false;
        mIndicator->mIsVisible = false;

        if( mParent )
            mLevel = mParent->mLevel + 1;
    }

    void UITreeViewNode::PushStyles()
    {
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2{ 0.0f, 3.0f } );
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2{ 0.0f, 0.0f } );
        ImGui::PushStyleColor( ImGuiCol_HeaderActive, ImVec4{ .025f, 0.025f, 0.025f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_HeaderHovered, ImVec4{ .025f, .025f, 0.025f, 1.0f } );
    }

    void UITreeViewNode::PopStyles()
    {
        ImGui::PopStyleVar( 2 );
        ImGui::PopStyleColor( 2 );
    }

    UITreeViewNode *UITreeViewNode::Add()
    {
        auto lNewChild = new UITreeViewNode( mTreeView, this );
        mChildren.push_back( lNewChild );

        return lNewChild;
    }

    void UITreeViewNode::OnSelected( std::function<void( UITreeViewNode * )> aOnSelected )
    {
        mOnSelected = aOnSelected;
    }

    void UITreeViewNode::SetIcon( UIImage *aIcon )
    {
        mIcon = aIcon;
    }

    void UITreeViewNode::SetIndicator( UIComponent *aIcon )
    {
        mIndicator->Add( aIcon, "IMAGE" );
        mIndicator->mIsVisible = !( aIcon == nullptr );
    }

    void UITreeViewNode::SetText( string_t const &aText )
    {
        mText->SetText( aText );
    }
    void UITreeViewNode::SetTextColor( math::vec4 aColor )
    {
        mText->SetTextColor( aColor );
    }

    ImVec2 UITreeViewNode::RequiredSize()
    {
        return mLayout->RequiredSize();
    }

    bool UITreeViewNode::IsOpen()
    {
        if( mParent == nullptr )
            return true;

        return mIsOpen;
    }

    bool UITreeViewNode::IsLeaf()
    {
        return mChildren.size() == 0;
    }

    void UITreeViewNode::RenderNode()
    {
        ImGuiWindow *lWindow = ImGui::GetCurrentWindow();

        ImGuiContext     &lImGuiContext = *GImGui;
        const ImGuiStyle &style         = lImGuiContext.Style;
        const ImVec2      lPadding =
            ( mFlags & ImGuiTreeNodeFlags_FramePadding )
                     ? style.FramePadding
                     : ImVec2( style.FramePadding.x, ImMin( lWindow->DC.CurrLineTextBaseOffset, style.FramePadding.y ) );

        const ImVec2 lLabelSize = mLayout->RequiredSize();

        // We vertically grow up to current line height up the typical widget height.
        const float lFrameHeight = ImMax( ImMin( lWindow->DC.CurrLineSize.y, lImGuiContext.FontSize + style.FramePadding.y * 2 ),
                                          lLabelSize.y + lPadding.y * 2 );

        ImRect lFrameBoundingBox;
        lFrameBoundingBox.Min.x = ( mFlags & ImGuiTreeNodeFlags_SpanFullWidth ) ? lWindow->WorkRect.Min.x : lWindow->DC.CursorPos.x;
        lFrameBoundingBox.Min.y = lWindow->DC.CursorPos.y;
        lFrameBoundingBox.Max.x = lWindow->WorkRect.Max.x;
        lFrameBoundingBox.Max.y = lWindow->DC.CursorPos.y + lFrameHeight;

        const float lTextOffsetX = lImGuiContext.FontSize + lPadding.x * 2 + mTreeView->mIconSpacing;
        const float lTextOffsetY = ImMax( lPadding.y, lWindow->DC.CurrLineTextBaseOffset );
        const float lTextWidth   = lImGuiContext.FontSize + ( lLabelSize.x > 0.0f ? lLabelSize.x + lPadding.x * 2 : 0.0f );
        ImVec2      lTextPosition( lWindow->DC.CursorPos.x + lTextOffsetX, lWindow->DC.CursorPos.y + lTextOffsetY );
        ImGui::ItemSize( ImVec2( lTextWidth, lFrameHeight ), lPadding.y );

        // For regular tree nodes, we arbitrary allow to click past 2 worth of ItemSpacing
        ImRect lInteractionBoundingBox = lFrameBoundingBox;
        if( ( mFlags & ( ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_SpanFullWidth ) ) == 0 )
            lInteractionBoundingBox.Max.x = lFrameBoundingBox.Min.x + lTextWidth + style.ItemSpacing.x * 2.0f;

        bool lNodeIsLeaf = IsLeaf();
        bool lNodeIsOpen = IsOpen();

        lImGuiContext.LastItemData.StatusFlags |= ImGuiItemStatusFlags_HasDisplayRect;
        lImGuiContext.LastItemData.DisplayRect = lFrameBoundingBox;

        ImGuiButtonFlags lButtonFlags = ImGuiTreeNodeFlags_None;

        if( mFlags & ImGuiTreeNodeFlags_AllowItemOverlap )
            lButtonFlags |= ImGuiButtonFlags_AllowItemOverlap;

        if( !lNodeIsLeaf )
            lButtonFlags |= ImGuiButtonFlags_PressedOnDragDropHold;

        const float lArrowHitX1 = ( lTextPosition.x - lTextOffsetX ) - style.TouchExtraPadding.x;
        const float lArrowHitX2 =
            ( lTextPosition.x - lTextOffsetX ) + ( lImGuiContext.FontSize + lPadding.x * 2.0f ) + style.TouchExtraPadding.x;
        const bool lIsMouseXOverArrow = ( lImGuiContext.IO.MousePos.x >= lArrowHitX1 && lImGuiContext.IO.MousePos.x < lArrowHitX2 );
        if( lWindow != lImGuiContext.HoveredWindow || !lIsMouseXOverArrow )
            lButtonFlags |= ImGuiButtonFlags_NoKeyModifiers;

        if( lIsMouseXOverArrow )
            lButtonFlags |= ImGuiButtonFlags_PressedOnClick;
        else if( mFlags & ImGuiTreeNodeFlags_OpenOnDoubleClick )
            lButtonFlags |= ImGuiButtonFlags_PressedOnClickRelease | ImGuiButtonFlags_PressedOnDoubleClick;
        else
            lButtonFlags |= ImGuiButtonFlags_PressedOnClickRelease;

        bool lSelected = ( mFlags & ImGuiTreeNodeFlags_Selected ) != 0;

        bool lIsHovered;
        bool lIsHeld;
        bool lIsPressed =
            ImGui::ButtonBehavior( lInteractionBoundingBox, lWindow->GetID( (void *)this ), &lIsHovered, &lIsHeld, lButtonFlags );

        bool lDoubleClicked = lIsHovered && ImGui::IsMouseDoubleClicked( 0 );

        bool lIsToggled = false;
        if( !lNodeIsLeaf )
        {
            if( lIsPressed && lImGuiContext.DragDropHoldJustPressedId != lWindow->GetID( (void *)this ) )
            {
                if( ( mFlags & ( ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick ) ) == 0 ||
                    ( lImGuiContext.NavActivateId == lWindow->GetID( (void *)this ) ) )
                    lIsToggled = true;
                if( mFlags & ImGuiTreeNodeFlags_OpenOnArrow )
                    lIsToggled |= lIsMouseXOverArrow && !lImGuiContext.NavDisableMouseHover;
                if( ( mFlags & ImGuiTreeNodeFlags_OpenOnDoubleClick ) && lImGuiContext.IO.MouseClickedCount[0] == 2 )
                    lIsToggled = true;
            }
            else if( lIsPressed && lImGuiContext.DragDropHoldJustPressedId == lWindow->GetID( (void *)this ) )
            {
                IM_ASSERT( lButtonFlags & ImGuiButtonFlags_PressedOnDragDropHold );
                if( !lNodeIsOpen ) // When using Drag and Drop "hold to open" we keep the node highlighted after opening, but never
                                   // close it again.
                    lIsToggled = true;
            }

            if( lImGuiContext.NavId == lWindow->GetID( (void *)this ) && lImGuiContext.NavMoveDir == ImGuiDir_Left && lNodeIsOpen )
            {
                lIsToggled = true;
                ImGui::NavMoveRequestCancel();
            }

            if( lImGuiContext.NavId == lWindow->GetID( (void *)this ) && lImGuiContext.NavMoveDir == ImGuiDir_Right &&
                !lNodeIsOpen ) // If there's something upcoming on the line we may want to give it the priority?
            {
                lIsToggled = true;
                ImGui::NavMoveRequestCancel();
            }

            if( lIsToggled )
            {
                lNodeIsOpen = !lNodeIsOpen;
                lWindow->DC.StateStorage->SetInt( lWindow->GetID( (void *)this ), lNodeIsOpen );
                lImGuiContext.LastItemData.StatusFlags |= ImGuiItemStatusFlags_ToggledOpen;
            }
        }

        if( mFlags & ImGuiTreeNodeFlags_AllowItemOverlap )
            ImGui::SetItemAllowOverlap();

        // Render
        const ImU32            lTextColor         = ImGui::GetColorU32( ImGuiCol_Text );
        const ImU32            lArrowColor        = IM_COL32( 60, 60, 60, 100 );
        ImGuiNavHighlightFlags lNavHighlightFlags = ImGuiNavHighlightFlags_TypeThin;

        if( lIsHovered || lSelected )
        {
            const ImU32 lBackgroundColor = ImGui::GetColorU32( ( lIsHeld && lIsHovered ) ? ImGuiCol_HeaderActive
                                                               : lIsHovered              ? ImGuiCol_HeaderHovered
                                                                                         : ImGuiCol_Header );
            ImGui::RenderFrame( lFrameBoundingBox.Min, lFrameBoundingBox.Max, lBackgroundColor, false );
        }

        ImGui::RenderNavHighlight( lFrameBoundingBox, lWindow->GetID( (void *)this ), lNavHighlightFlags );
        if( !lNodeIsLeaf )
            RenderArrow(
                lWindow->DrawList,
                ImVec2( lTextPosition.x - lTextOffsetX + ( lTextOffsetX - lWindow->DrawList->_Data->FontSize * .75f * 0.8f ) * 0.5f +
                            mTreeView->mIndent * mLevel,
                        lTextPosition.y + ( lFrameHeight - style.FramePadding.y - lWindow->DrawList->_Data->FontSize * 0.8f ) * 0.5f ),
                lArrowColor, lNodeIsOpen ? ImGuiDir_Down : ImGuiDir_Right, 0.8f );
        else
            RenderIcon(
                lWindow->DrawList,
                ImVec2( lTextPosition.x - lTextOffsetX + ( lTextOffsetX - lWindow->DrawList->_Data->FontSize * .75f * 0.8f ) * 0.5f +
                            mTreeView->mIndent * mLevel,
                        lTextPosition.y + ( lFrameHeight - style.FramePadding.y - lWindow->DrawList->_Data->FontSize ) * 0.5f ) );

        ImVec2 lSize{ lWindow->WorkRect.Max.x - lWindow->WorkRect.Min.x, lFrameHeight };

        auto lNodePosition = ImGui::GetCursorPos() + ImVec2{ lTextOffsetX + mTreeView->mIndent * mLevel, -lFrameHeight };
        mLayout->Update( lNodePosition, lSize );

        if( lDoubleClicked && lNodeIsLeaf && mOnSelected )
            mOnSelected( this );

        if( lNodeIsOpen != mIsOpen )
        {
            mIsOpen = lNodeIsOpen;
            mTreeView->UpdateRows();
        }
    }

    void UITreeViewNode::RenderArrow( ImDrawList *aDrawList, ImVec2 aPosition, ImU32 aColor, ImGuiDir aDirection, float aScale )
    {
        const float lHeight = aDrawList->_Data->FontSize * .75f;
        float       lRadius = lHeight * 0.5f * aScale;
        ImVec2      lCenter = aPosition + ImVec2( lHeight * 0.50f, lHeight * 0.50f * aScale );

        ImVec2 a, b, c;
        switch( aDirection )
        {
        case ImGuiDir_Up:
        case ImGuiDir_Down:
            if( aDirection == ImGuiDir_Up )
                lRadius = -lRadius;
            a = ImVec2( +0.000f, +0.750f ) * lRadius;
            b = ImVec2( -0.866f, -0.750f ) * lRadius;
            c = ImVec2( +0.866f, -0.750f ) * lRadius;
            break;
        case ImGuiDir_Left:
        case ImGuiDir_Right:
            if( aDirection == ImGuiDir_Left )
                lRadius = -lRadius;
            a = ImVec2( +0.750f, +0.000f ) * lRadius;
            b = ImVec2( -0.750f, +0.866f ) * lRadius;
            c = ImVec2( -0.750f, -0.866f ) * lRadius;
            break;
        case ImGuiDir_None:
        case ImGuiDir_COUNT:
            IM_ASSERT( 0 );
            break;
        }

        aDrawList->AddTriangleFilled( lCenter + a, lCenter + b, lCenter + c, aColor );
    }

    void UITreeViewNode::RenderIcon( ImDrawList *aDrawList, ImVec2 aPosition )
    {
        const float lHeight = aDrawList->_Data->FontSize;

        if( mImage != nullptr )
        {
            aDrawList->AddImage( mIcon->TextureID(), aPosition, aPosition + ImVec2{ lHeight, lHeight }, ImVec2{ 0, 0 },
                                 ImVec2{ 1, 1 } );
        }
    }

    vector_t<UITreeViewNode *> const &UITreeViewNode::Children()
    {
        return mChildren;
    }

    void UITreeViewNode::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        RenderNode();
    }

    UITreeView::UITreeView()
    {
        SetIndent( 9.0f );
        mRoot = new UITreeViewNode( this, nullptr );
    }

    void UITreeView::PushStyles()
    {
    }

    void UITreeView::PopStyles()
    {
    }

    ImVec2 UITreeView::RequiredSize()
    {
        return ImVec2{};
    }

    void UITreeView::SetIndent( float aIndent )
    {
        mIndent = aIndent;
    }

    void UITreeView::SetIconSpacing( float aSpacing )
    {
        mIconSpacing = aSpacing;
    }

    UITreeViewNode *UITreeView::Add()
    {
        return mRoot->Add();
    }

    void UITreeView::UpdateRows()
    {
        mRows.clear();

        vec_t<UITreeViewNode *> lStack;

        auto lRootChildren = vec_t( mRoot->Children().begin(), mRoot->Children().end() );
        std::reverse( lRootChildren.begin(), lRootChildren.end() );
        for( auto *lNode : lRootChildren )
            lStack.push_back( lNode );

        while( lStack.size() > 0 )
        {
            auto *lNode = lStack.back();
            lStack.pop_back();
            mRows.push_back( lNode );

            if( !lNode->IsLeaf() && lNode->IsOpen() )
            {
                auto lChildren = vec_t( lNode->Children().begin(), lNode->Children().end() );
                std::reverse( lChildren.begin(), lChildren.end() );

                for( auto *lChild : lChildren )
                    lStack.push_back( lChild );
            }
        }
    }

    void UITreeView::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( aPosition );

        const ImGuiTableFlags lTableFlags =
            ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV;

        ImGui::PushID( (void *)this );
        ImGui::BeginChild( "##ContainerItem", aSize );

        ImGuiListClipper lRowClipping;
        lRowClipping.Begin( mRows.size(), 20.0f );

        while( lRowClipping.Step() )
        {
            for( int lRowID = lRowClipping.DisplayStart; lRowID < lRowClipping.DisplayEnd; lRowID++ )
            {
                if( lRowID >= mRows.size() )
                    continue;

                ImVec2 lPos = ImGui::GetCursorPos();
                mRows[lRowID]->Update( lPos, ImVec2{ aSize.x, 20.0f } );
                lPos.y += 20.0f;
                ImGui::SetCursorPos( lPos );
            }
        }

        ImGui::EndChild();
        ImGui::PopID();
    }
} // namespace SE::Core