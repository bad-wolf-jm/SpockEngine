
#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

namespace SE::Core::UI
{
    bool TreeNodeBehavior( ImGuiID aID, ImGuiTreeNodeFlags aFlags, const char *aLabel, const char *aLabelEnd );
    void TreePushOverrideID( ImGuiID aID );

    bool TreeNode( const char *aLabel )
    {
        ImGuiWindow *lWindow = ImGui::GetCurrentWindow();
        if( lWindow->SkipItems ) return false;
        return TreeNodeBehavior( lWindow->GetID( aLabel ), 0, aLabel, NULL );
    }

    bool TreeNodeEx( const char *aLabel, ImGuiTreeNodeFlags aFlags )
    {
        ImGuiWindow *lWindow = ImGui::GetCurrentWindow();
        if( lWindow->SkipItems ) return false;

        return TreeNodeBehavior( lWindow->GetID( aLabel ), aFlags, aLabel, NULL );
    }

    bool TreeNodeBehaviorIsOpen( ImGuiID aID, ImGuiTreeNodeFlags aFlags )
    {
        if( aFlags & ImGuiTreeNodeFlags_Leaf ) return true;

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
                lWindowStorage->SetInt( aID, lNodeIsOpen );
            }
            else
            {
                // We treat ImGuiCond_Once and ImGuiCond_FirstUseEver the same because tree node state are not saved persistently.
                const int stored_value = lWindowStorage->GetInt( aID, -1 );
                if( stored_value == -1 )
                {
                    lNodeIsOpen = lImGuiContext.NextItemData.OpenVal;
                    lWindowStorage->SetInt( aID, lNodeIsOpen );
                }
                else
                {
                    lNodeIsOpen = stored_value != 0;
                }
            }
        }
        else
        {
            lNodeIsOpen = lWindowStorage->GetInt( aID, ( aFlags & ImGuiTreeNodeFlags_DefaultOpen ) ? 1 : 0 ) != 0;
        }

        // When logging is enabled, we automatically expand tree nodes (but *NOT* collapsing headers.. seems like sensible behavior).
        // NB- If we are above max depth we still allow manually opened nodes to be logged.
        if( lImGuiContext.LogEnabled && !( aFlags & ImGuiTreeNodeFlags_NoAutoOpenOnLog ) &&
            ( lWindow->DC.TreeDepth - lImGuiContext.LogDepthRef ) < lImGuiContext.LogDepthToExpand )
            lNodeIsOpen = true;

        return lNodeIsOpen;
    }

    void RenderArrow( ImDrawList *aDrawList, ImVec2 aPosition, ImU32 aColor, ImGuiDir aDirection, float aScale )
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
        case ImGuiDir_COUNT:
            IM_ASSERT( 0 );
            break;
        }
        aDrawList->AddTriangleFilled( lCenter + a, lCenter + b, lCenter + c, aColor );
    }

    bool TreeNodeBehavior( ImGuiID aID, ImGuiTreeNodeFlags aFlags, const char *aLabel, const char *aLabelEnd )
    {
        ImGuiWindow *lWindow = ImGui::GetCurrentWindow();
        if( lWindow->SkipItems ) return false;

        ImGuiContext     &lImGuiContext = *GImGui;
        const ImGuiStyle &style         = lImGuiContext.Style;
        const ImVec2      lPadding =
            ( aFlags & ImGuiTreeNodeFlags_FramePadding )
                     ? style.FramePadding
                     : ImVec2( style.FramePadding.x, ImMin( lWindow->DC.CurrLineTextBaseOffset, style.FramePadding.y ) );

        if( !aLabelEnd ) aLabelEnd = ImGui::FindRenderedTextEnd( aLabel );
        const ImVec2 lLabelSize = ImGui::CalcTextSize( aLabel, aLabelEnd, false );

        // We vertically grow up to current line height up the typical widget height.
        const float lFrameHeight = ImMax(
            ImMin( lWindow->DC.CurrLineSize.y, lImGuiContext.FontSize + style.FramePadding.y * 2 ), lLabelSize.y + lPadding.y * 2 );
        ImRect lFrameBoundingBox;
        lFrameBoundingBox.Min.x = ( aFlags & ImGuiTreeNodeFlags_SpanFullWidth ) ? lWindow->WorkRect.Min.x : lWindow->DC.CursorPos.x;
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
        if( ( aFlags & ( ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_SpanFullWidth ) ) == 0 )
            lInteractionBoundingBox.Max.x = lFrameBoundingBox.Min.x + lTextWidth + style.ItemSpacing.x * 2.0f;

        // Store a flag for the current depth to tell if we will allow closing this node when navigating one of its child.
        // For this purpose we essentially compare if lImGuiContext.NavIdIsAlive went from 0 to 1 between TreeNode() and TreePop().
        // This is currently only support 32 level deep and we are fine with (1 << Depth) overflowing into a zero.
        const bool lNodeIsLeaf = ( aFlags & ImGuiTreeNodeFlags_Leaf ) != 0;
        bool       lNodeIsOpen = TreeNodeBehaviorIsOpen( aID, aFlags );
        if( lNodeIsOpen && !lImGuiContext.NavIdIsAlive && ( aFlags & ImGuiTreeNodeFlags_NavLeftJumpsBackHere ) &&
            !( aFlags & ImGuiTreeNodeFlags_NoTreePushOnOpen ) )
            lWindow->DC.TreeJumpToParentOnPopMask |= ( 1 << lWindow->DC.TreeDepth );

        bool lItemAdd = ImGui::ItemAdd( lInteractionBoundingBox, aID );
        lImGuiContext.LastItemData.StatusFlags |= ImGuiItemStatusFlags_HasDisplayRect;
        lImGuiContext.LastItemData.DisplayRect = lFrameBoundingBox;

        if( !lItemAdd )
        {
            if( lNodeIsOpen && !( aFlags & ImGuiTreeNodeFlags_NoTreePushOnOpen ) ) TreePushOverrideID( aID );

            return lNodeIsOpen;
        }

        ImGuiButtonFlags lButtonFlags = ImGuiTreeNodeFlags_None;
        if( aFlags & ImGuiTreeNodeFlags_AllowItemOverlap ) lButtonFlags |= ImGuiButtonFlags_AllowItemOverlap;
        if( !lNodeIsLeaf ) lButtonFlags |= ImGuiButtonFlags_PressedOnDragDropHold;

        // We allow clicking on the arrow section with keyboard modifiers held, in order to easily
        // allow browsing a tree while preserving selection with code implementing multi-selection patterns.
        // When clicking on the rest of the tree node we always disallow keyboard modifiers.
        const float lArrowHitX1 = ( lTextPosition.x - lTextOffsetX ) - style.TouchExtraPadding.x;
        const float lArrowHitX2 =
            ( lTextPosition.x - lTextOffsetX ) + ( lImGuiContext.FontSize + lPadding.x * 2.0f ) + style.TouchExtraPadding.x;
        const bool lIsMouseXOverArrow = ( lImGuiContext.IO.MousePos.x >= lArrowHitX1 && lImGuiContext.IO.MousePos.x < lArrowHitX2 );
        if( lWindow != lImGuiContext.HoveredWindow || !lIsMouseXOverArrow ) lButtonFlags |= ImGuiButtonFlags_NoKeyModifiers;

        // Open behaviors can be altered with the _OpenOnArrow and _OnOnDoubleClick aFlags.
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
        else if( aFlags & ImGuiTreeNodeFlags_OpenOnDoubleClick )
            lButtonFlags |= ImGuiButtonFlags_PressedOnClickRelease | ImGuiButtonFlags_PressedOnDoubleClick;
        else
            lButtonFlags |= ImGuiButtonFlags_PressedOnClickRelease;

        bool       lSelected    = ( aFlags & ImGuiTreeNodeFlags_Selected ) != 0;
        const bool lWasSelected = lSelected;

        bool lIsHovered, lIsHeld;
        bool lIsPressed = ImGui::ButtonBehavior( lInteractionBoundingBox, aID, &lIsHovered, &lIsHeld, lButtonFlags );
        bool lIsToggled = false;
        if( !lNodeIsLeaf )
        {
            if( lIsPressed && lImGuiContext.DragDropHoldJustPressedId != aID )
            {
                if( ( aFlags & ( ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick ) ) == 0 ||
                    ( lImGuiContext.NavActivateId == aID ) )
                    lIsToggled = true;
                if( aFlags & ImGuiTreeNodeFlags_OpenOnArrow )
                    lIsToggled |=
                        lIsMouseXOverArrow && !lImGuiContext.NavDisableMouseHover; // Lightweight equivalent of IsMouseHoveringRect()
                                                                                   // since ImGui::ButtonBehavior() already did the job
                if( ( aFlags & ImGuiTreeNodeFlags_OpenOnDoubleClick ) && lImGuiContext.IO.MouseClickedCount[0] == 2 )
                    lIsToggled = true;
            }
            else if( lIsPressed && lImGuiContext.DragDropHoldJustPressedId == aID )
            {
                IM_ASSERT( lButtonFlags & ImGuiButtonFlags_PressedOnDragDropHold );
                if( !lNodeIsOpen ) // When using Drag and Drop "hold to open" we keep the node highlighted after opening, but never
                                   // close it again.
                    lIsToggled = true;
            }

            if( lImGuiContext.NavId == aID && lImGuiContext.NavMoveDir == ImGuiDir_Left && lNodeIsOpen )
            {
                lIsToggled = true;
                ImGui::NavMoveRequestCancel();
            }
            if( lImGuiContext.NavId == aID && lImGuiContext.NavMoveDir == ImGuiDir_Right &&
                !lNodeIsOpen ) // If there's something upcoming on the line we may want to give it the priority?
            {
                lIsToggled = true;
                ImGui::NavMoveRequestCancel();
            }

            if( lIsToggled )
            {
                lNodeIsOpen = !lNodeIsOpen;
                lWindow->DC.StateStorage->SetInt( aID, lNodeIsOpen );
                lImGuiContext.LastItemData.StatusFlags |= ImGuiItemStatusFlags_ToggledOpen;
            }
        }
        if( aFlags & ImGuiTreeNodeFlags_AllowItemOverlap ) ImGui::SetItemAllowOverlap();

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
        ImGui::RenderNavHighlight( lFrameBoundingBox, aID, lNavHighlightFlags );
        if( aFlags & ImGuiTreeNodeFlags_Bullet )
            ImGui::RenderBullet( lWindow->DrawList,
                ImVec2( lTextPosition.x - lTextOffsetX * 0.5f, lTextPosition.y + lImGuiContext.FontSize * 0.5f ), lArrowColor );
        else if( !lNodeIsLeaf )
            RenderArrow( lWindow->DrawList,
                ImVec2( lTextPosition.x - lTextOffsetX + lPadding.x, lTextPosition.y + lImGuiContext.FontSize * 0.16f ), lArrowColor,
                lNodeIsOpen ? ImGuiDir_Down : ImGuiDir_Right, 0.8f );
        if( lImGuiContext.LogEnabled ) ImGui::LogSetNextTextDecoration( ">", NULL );
        ImGui::RenderText( lTextPosition, aLabel, aLabelEnd, false );

        if( lNodeIsOpen && !( aFlags & ImGuiTreeNodeFlags_NoTreePushOnOpen ) ) TreePushOverrideID( aID );

        return lNodeIsOpen;
    }

    void TreePushOverrideID( ImGuiID aID )
    {
        ImGuiContext &lImGuiContext = *GImGui;
        ImGuiWindow  *lWindow       = lImGuiContext.CurrentWindow;
        ImGui::Indent( lImGuiContext.FontSize / 1.5f );
        lWindow->DC.TreeDepth++;
        ImGui::PushOverrideID( aID );
    }

    void TreePop()
    {
        ImGuiContext &lImGuiContext = *GImGui;
        ImGuiWindow  *lWindow       = lImGuiContext.CurrentWindow;
        ImGui::Unindent( lImGuiContext.FontSize / 1.5f );

        lWindow->DC.TreeDepth--;
        ImU32 tree_depth_mask = ( 1 << lWindow->DC.TreeDepth );

        // Handle Left arrow to move to parent tree node (when ImGuiTreeNodeFlags_NavLeftJumpsBackHere is enabled)
        if( lImGuiContext.NavMoveDir == ImGuiDir_Left && lImGuiContext.NavWindow == lWindow && ImGui::NavMoveRequestButNoResultYet() )
            if( lImGuiContext.NavIdIsAlive && ( lWindow->DC.TreeJumpToParentOnPopMask & tree_depth_mask ) )
            {
                ImGui::SetNavID( lWindow->IDStack.back(), lImGuiContext.NavLayer, 0, ImRect() );
                ImGui::NavMoveRequestCancel();
            }
        lWindow->DC.TreeJumpToParentOnPopMask &= tree_depth_mask - 1;

        IM_ASSERT( lWindow->IDStack.Size > 1 ); // There should always be 1 element in the IDStack (pushed during lWindow creation). If
                                                // this triggers you called TreePop/ImGui::PopID too much.
        ImGui::PopID();
    }
} // namespace SE::Core::UI