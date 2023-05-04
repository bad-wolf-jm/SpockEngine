#include "TreeView.h"

#include "DotNet/Runtime.h"

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

namespace SE::Core
{
    UITreeViewNode::UITreeViewNode( std::string const &aText )
        : mText{ aText }
    {
    }

    void UITreeViewNode::PushStyles() {}
    void UITreeViewNode::PopStyles() {}

    void UITreeViewNode::SetText( std::string const &aText ) { mText = aText; }
    void UITreeViewNode::SetTextColor( math::vec4 aColor ) { mTextColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }

    ImVec2 UITreeViewNode::RequiredSize() { return mNodeLayout->RequiredSize(); }

    void UITreeViewNode::TreePushOverrideID()
    {
        ImGuiContext &lImGuiContext = *GImGui;
        lImGuiContext.CurrentWindow->DC.TreeDepth++;

        ImGui::Indent( mIndent );
        ImGui::PushOverrideID( mID );
    }

    void UITreeViewNode::TreePop()
    {
        ImGuiContext &lImGuiContext = *GImGui;
        ImGuiWindow  *lWindow       = GImGui->CurrentWindow;

        ImGui::Unindent( mIndent );

        lWindow->DC.TreeDepth--;
        ImU32 lTreeDepthMask = ( 1 << lWindow->DC.TreeDepth );

        // Handle Left arrow to move to parent tree node (when ImGuiTreeNodeFlags_NavLeftJumpsBackHere is enabled)
        if( lImGuiContext.NavMoveDir == ImGuiDir_Left && lImGuiContext.NavWindow == lWindow && ImGui::NavMoveRequestButNoResultYet() )
        {
            if( lImGuiContext.NavIdIsAlive && ( lWindow->DC.TreeJumpToParentOnPopMask & lTreeDepthMask ) )
            {
                ImGui::SetNavID( lWindow->IDStack.back(), lImGuiContext.NavLayer, 0, ImRect() );
                ImGui::NavMoveRequestCancel();
            }
        }

        lWindow->DC.TreeJumpToParentOnPopMask &= lTreeDepthMask - 1;

        IM_ASSERT( lWindow->IDStack.Size > 1 );

        ImGui::PopID();
    }

    bool UITreeViewNode::IsOpen()
    {
        if( mFlags & ImGuiTreeNodeFlags_Leaf ) return true;

        // We only write to the tree storage if the user clicks (or explicitly use the SetNextItemOpen function)
        ImGuiContext &lImGuiContext  = *GImGui;
        ImGuiWindow  *lWindow        = lImGuiContext.CurrentWindow;
        ImGuiStorage *lWindowStorage = lWindow->DC.StateStorage;

        bool lNodeIsOpen;
        if( lImGuiContext.NextItemData.Flags & ImGuiNextItemDataFlags_HasOpen )
        {
            if( lImGuiContext.NextItemData.OpenCond & ImGuiCond_Always )
            {
                lNodeIsOpen = lImGuiContext.NextItemData.OpenVal;
                lWindowStorage->SetInt( mID, lNodeIsOpen );
            }
            else
            {
                // We treat ImGuiCond_Once and ImGuiCond_FirstUseEver the same because tree node state are not saved persistently.
                const int lStoredValue = lWindowStorage->GetInt( mID, -1 );
                if( lStoredValue == -1 )
                {
                    lNodeIsOpen = lImGuiContext.NextItemData.OpenVal;
                    lWindowStorage->SetInt( mID, lNodeIsOpen );
                }
                else
                {
                    lNodeIsOpen = ( lStoredValue != 0 );
                }
            }
        }
        else
        {
            lNodeIsOpen = lWindowStorage->GetInt( mID, ( mFlags & ImGuiTreeNodeFlags_DefaultOpen ) ? 1 : 0 ) != 0;
        }

        return lNodeIsOpen;
    }

    bool UITreeViewNode::IsLeaf() { return mChildren.size() == 0; }

    bool UITreeViewNode::RenderNode()
    {
        ImGuiWindow *lWindow = ImGui::GetCurrentWindow();
        if( lWindow->SkipItems ) return false;

        ImGuiContext     &lImGuiContext = *GImGui;
        const ImGuiStyle &style         = lImGuiContext.Style;
        const ImVec2      lPadding =
            ( mFlags & ImGuiTreeNodeFlags_FramePadding )
                     ? style.FramePadding
                     : ImVec2( style.FramePadding.x, ImMin( lWindow->DC.CurrLineTextBaseOffset, style.FramePadding.y ) );

        if( !aLabelEnd ) aLabelEnd = ImGui::FindRenderedTextEnd( aLabel );
        const ImVec2 lLabelSize = ImGui::CalcTextSize( aLabel, aLabelEnd, false );

        // We vertically grow up to current line height up the typical widget height.
        const float lFrameHeight = ImMax( ImMin( lWindow->DC.CurrLineSize.y, lImGuiContext.FontSize + style.FramePadding.y * 2 ),
                                          lLabelSize.y + lPadding.y * 2 );
        ImRect      lFrameBoundingBox;
        lFrameBoundingBox.Min.x = ( mFlags & ImGuiTreeNodeFlags_SpanFullWidth ) ? lWindow->WorkRect.Min.x : lWindow->DC.CursorPos.x;
        lFrameBoundingBox.Min.y = lWindow->DC.CursorPos.y;
        lFrameBoundingBox.Max.x = lWindow->WorkRect.Max.x;
        lFrameBoundingBox.Max.y = lWindow->DC.CursorPos.y + lFrameHeight;

        const float lTextOffsetX = lImGuiContext.FontSize + lPadding.x * 2;
        const float lTextOffsetY =
            ImMax( lPadding.y, lWindow->DC.CurrLineTextBaseOffset ); // Latch before ImGui::CalcTextSize changes it
        const float lTextWidth =
            lImGuiContext.FontSize + ( lLabelSize.x > 0.0f ? lLabelSize.x + lPadding.x * 2 : 0.0f ); // Include collapser
        ImVec2 lTextPosition( lWindow->DC.CursorPos.x + lTextOffsetX, lWindow->DC.CursorPos.y + lTextOffsetY );
        ImGui::ItemSize( ImVec2( lTextWidth, lFrameHeight ), lPadding.y );

        // For regular tree nodes, we arbitrary allow to click past 2 worth of ItemSpacing
        ImRect lInteractionBoundingBox = lFrameBoundingBox;
        if( ( mFlags & ( ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_SpanFullWidth ) ) == 0 )
            lInteractionBoundingBox.Max.x = lFrameBoundingBox.Min.x + lTextWidth + style.ItemSpacing.x * 2.0f;

        // Store a flag for the current depth to tell if we will allow closing this node when navigating one of its child.
        // For this purpose we essentially compare if lImGuiContext.NavIdIsAlive went from 0 to 1 between TreeNode() and TreePop().
        // This is currently only support 32 level deep and we are fine with (1 << Depth) overflowing into a zero.
        const bool lNodeIsLeaf = ( mFlags & ImGuiTreeNodeFlags_Leaf ) != 0;
        bool       lNodeIsOpen = TreeNodeBehaviorIsOpen( mID, mFlags );
        if( lNodeIsOpen && !lImGuiContext.NavIdIsAlive && ( mFlags & ImGuiTreeNodeFlags_NavLeftJumpsBackHere ) &&
            !( mFlags & ImGuiTreeNodeFlags_NoTreePushOnOpen ) )
            lWindow->DC.TreeJumpToParentOnPopMask |= ( 1 << lWindow->DC.TreeDepth );

        bool lItemAdd = ImGui::ItemAdd( lInteractionBoundingBox, mID );
        lImGuiContext.LastItemData.StatusFlags |= ImGuiItemStatusFlags_HasDisplayRect;
        lImGuiContext.LastItemData.DisplayRect = lFrameBoundingBox;

        if( !lItemAdd )
        {
            if( lNodeIsOpen && !( mFlags & ImGuiTreeNodeFlags_NoTreePushOnOpen ) ) TreePushOverrideID( mID );

            return lNodeIsOpen;
        }

        ImGuiButtonFlags lButtonFlags = ImGuiTreeNodeFlags_None;
        if( mFlags & ImGuiTreeNodeFlags_AllowItemOverlap ) lButtonFlags |= ImGuiButtonFlags_AllowItemOverlap;
        if( !lNodeIsLeaf ) lButtonFlags |= ImGuiButtonFlags_PressedOnDragDropHold;

        // We allow clicking on the arrow section with keyboard modifiers held, in order to easily
        // allow browsing a tree while preserving selection with code implementing multi-selection patterns.
        // When clicking on the rest of the tree node we always disallow keyboard modifiers.
        const float lArrowHitX1 = ( lTextPosition.x - lTextOffsetX ) - style.TouchExtraPadding.x;
        const float lArrowHitX2 =
            ( lTextPosition.x - lTextOffsetX ) + ( lImGuiContext.FontSize + lPadding.x * 2.0f ) + style.TouchExtraPadding.x;
        const bool lIsMouseXOverArrow = ( lImGuiContext.IO.MousePos.x >= lArrowHitX1 && lImGuiContext.IO.MousePos.x < lArrowHitX2 );
        if( lWindow != lImGuiContext.HoveredWindow || !lIsMouseXOverArrow ) lButtonFlags |= ImGuiButtonFlags_NoKeyModifiers;

        // Open behaviors can be altered with the _OpenOnArrow and _OnOnDoubleClick mFlags.
        // Some alteration have subtle effects (e.lImGuiContext. toggle on MouseUp vs MouseDown events) due to requirements for
        // multi-selection and drag and drop support.
        // - Single-click on aLabel = Toggle on MouseUp (default, when _OpenOnArrow=0)
        // - Single-click on arrow = Toggle on MouseDown (when _OpenOnArrow=0)
        // - Single-click on arrow = Toggle on MouseDown (when _OpenOnArrow=1)
        // - Double-click on aLabel = Toggle on MouseDoubleClick (when _OpenOnDoubleClick=1)
        // - Double-click on arrow = Toggle on MouseDoubleClick (when _OpenOnDoubleClick=1 and _OpenOnArrow=0)
        // It is rather standard that arrow click react on Down rather than Up.
        // We set ImGuiButtonFlags_PressedOnClickRelease on OpenOnDoubleClick because we want the item to be active on the initial
        // MouseDown in order for drag and drop to work.
        if( lIsMouseXOverArrow )
            lButtonFlags |= ImGuiButtonFlags_PressedOnClick;
        else if( mFlags & ImGuiTreeNodeFlags_OpenOnDoubleClick )
            lButtonFlags |= ImGuiButtonFlags_PressedOnClickRelease | ImGuiButtonFlags_PressedOnDoubleClick;
        else
            lButtonFlags |= ImGuiButtonFlags_PressedOnClickRelease;

        bool       lSelected    = ( mFlags & ImGuiTreeNodeFlags_Selected ) != 0;
        const bool lWasSelected = lSelected;

        bool lIsHovered, lIsHeld;
        bool lIsPressed = ImGui::ButtonBehavior( lInteractionBoundingBox, mID, &lIsHovered, &lIsHeld, lButtonFlags );
        bool lIsToggled = false;
        if( !lNodeIsLeaf )
        {
            if( lIsPressed && lImGuiContext.DragDropHoldJustPressedId != mID )
            {
                if( ( mFlags & ( ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick ) ) == 0 ||
                    ( lImGuiContext.NavActivateId == mID ) )
                    lIsToggled = true;
                if( mFlags & ImGuiTreeNodeFlags_OpenOnArrow )
                    lIsToggled |=
                        lIsMouseXOverArrow && !lImGuiContext.NavDisableMouseHover; // Lightweight equivalent of IsMouseHoveringRect()
                                                                                   // since ImGui::ButtonBehavior() already did the job
                if( ( mFlags & ImGuiTreeNodeFlags_OpenOnDoubleClick ) && lImGuiContext.IO.MouseClickedCount[0] == 2 )
                    lIsToggled = true;
            }
            else if( lIsPressed && lImGuiContext.DragDropHoldJustPressedId == mID )
            {
                IM_ASSERT( lButtonFlags & ImGuiButtonFlags_PressedOnDragDropHold );
                if( !lNodeIsOpen ) // When using Drag and Drop "hold to open" we keep the node highlighted after opening, but never
                                   // close it again.
                    lIsToggled = true;
            }

            if( lImGuiContext.NavId == mID && lImGuiContext.NavMoveDir == ImGuiDir_Left && lNodeIsOpen )
            {
                lIsToggled = true;
                ImGui::NavMoveRequestCancel();
            }
            if( lImGuiContext.NavId == mID && lImGuiContext.NavMoveDir == ImGuiDir_Right &&
                !lNodeIsOpen ) // If there's something upcoming on the line we may want to give it the priority?
            {
                lIsToggled = true;
                ImGui::NavMoveRequestCancel();
            }

            if( lIsToggled )
            {
                lNodeIsOpen = !lNodeIsOpen;
                lWindow->DC.StateStorage->SetInt( mID, lNodeIsOpen );
                lImGuiContext.LastItemData.StatusFlags |= ImGuiItemStatusFlags_ToggledOpen;
            }
        }
        if( mFlags & ImGuiTreeNodeFlags_AllowItemOverlap ) ImGui::SetItemAllowOverlap();

        // In this branch, TreeNodeBehavior() cannot toggle the selection so this will never trigger.
        if( lSelected != lWasSelected ) //-V547
            lImGuiContext.LastItemData.StatusFlags |= ImGuiItemStatusFlags_ToggledSelection;

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
        ImGui::RenderNavHighlight( lFrameBoundingBox, mID, lNavHighlightFlags );
        if( !lNodeIsLeaf )
            RenderArrow( lWindow->DrawList,
                         ImVec2( lTextPosition.x - lTextOffsetX + lPadding.x, lTextPosition.y + lImGuiContext.FontSize * 0.16f ),
                         lArrowColor, lNodeIsOpen ? ImGuiDir_Down : ImGuiDir_Right, 0.8f );

        ImVec2 lSize{ lWindow->WorkRect.Max.x - lWindow->WorkRect.Min.x, lFrameHeight };
        mNodeLayout->Update( lTextPosition, lSize );

        if( lNodeIsOpen && !( mFlags & ImGuiTreeNodeFlags_NoTreePushOnOpen ) ) TreePushOverrideID( mID );

        return lNodeIsOpen;
    }

    void UITreeViewNode::RenderArrow( ImDrawList *aDrawList, ImVec2 aPosition, ImU32 aColor, ImGuiDir aDirection, float aScale )
    {
        const float lHeight = aDrawList->_Data->FontSize * 1.00f;
        float       lRadius = lHeight * 0.45f * aScale;
        ImVec2      lCenter = aPosition + ImVec2( lHeight * 0.50f, lHeight * 0.50f * aScale );

        ImVec2 a, b, c;
        switch( aDirection )
        {
        case ImGuiDir_Up:
        case ImGuiDir_Down:
            if( aDirection == ImGuiDir_Up ) lRadius = -lRadius;
            a = ImVec2( +0.000f, +0.750f ) * lRadius;
            b = ImVec2( -0.866f, -0.750f ) * lRadius;
            c = ImVec2( +0.866f, -0.750f ) * lRadius;
            break;
        case ImGuiDir_Left:
        case ImGuiDir_Right:
            if( aDirection == ImGuiDir_Left ) lRadius = -lRadius;
            a = ImVec2( +0.750f, +0.000f ) * lRadius;
            b = ImVec2( -0.750f, +0.866f ) * lRadius;
            c = ImVec2( -0.750f, -0.866f ) * lRadius;
            break;
        case ImGuiDir_None:
        case ImGuiDir_COUNT: IM_ASSERT( 0 ); break;
        }
        aDrawList->AddTriangleFilled( lCenter + a, lCenter + b, lCenter + c, aColor );
    }

    void UITreeViewNode::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet ) ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        auto lTextSize     = ImGui::CalcTextSize( mText.c_str() );
        auto lTextPosition = GetContentAlignedposition( mHAlign, mVAlign, aPosition, lTextSize, aSize );

        ImGui::SetCursorPos( aPosition );
        if( RenderNode() )
        {
            for( auto lChild : mChildren ) lChild->Update( ImGui::GetCursorPos(), aSize );

            TreePop();
        }
        if( lTextColorSet ) ImGui::PopStyleColor();
    }

    void *UITreeViewNode::UITreeViewNode_Create()
    {
        auto lNewLabel = new UITreeViewNode();

        return static_cast<void *>( lNewLabel );
    }

    void *UITreeViewNode::UITreeViewNode_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UITreeViewNode( lString );

        return static_cast<void *>( lNewLabel );
    }

    void UITreeViewNode::UITreeViewNode_Destroy( void *aInstance ) { delete static_cast<UITreeViewNode *>( aInstance ); }

    void UITreeViewNode::UITreeViewNode_SetText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UITreeViewNode::UITreeViewNode_SetTextColor( void *aInstance, math::vec4 *aTextColor )
    {
        auto lInstance = static_cast<UITreeViewNode *>( aInstance );

        lInstance->SetTextColor( *aTextColor );
    }
} // namespace SE::Core