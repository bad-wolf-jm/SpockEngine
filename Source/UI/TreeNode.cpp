
#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

namespace LTSE::Core::UI
{
    bool TreeNodeBehavior( ImGuiID id, ImGuiTreeNodeFlags flags, const char *label, const char *label_end );
    void TreePushOverrideID( ImGuiID id );

    bool TreeNode( const char *label )
    {
        ImGuiWindow *window = ImGui::GetCurrentWindow();
        if( window->SkipItems ) return false;
        return TreeNodeBehavior( window->GetID( label ), 0, label, NULL );
    }

    bool TreeNodeEx( const char *label, ImGuiTreeNodeFlags flags )
    {
        ImGuiWindow *window = ImGui::GetCurrentWindow();
        if( window->SkipItems ) return false;

        return TreeNodeBehavior( window->GetID( label ), flags, label, NULL );
    }

    bool TreeNodeBehaviorIsOpen( ImGuiID id, ImGuiTreeNodeFlags flags )
    {
        if( flags & ImGuiTreeNodeFlags_Leaf ) return true;

        // We only write to the tree storage if the user clicks (or explicitly use the SetNextItemOpen function)
        ImGuiContext &g       = *GImGui;
        ImGuiWindow  *window  = g.CurrentWindow;
        ImGuiStorage *storage = window->DC.StateStorage;

        bool is_open;
        if( g.NextItemData.Flags & ImGuiNextItemDataFlags_HasOpen )
        {
            if( g.NextItemData.OpenCond & ImGuiCond_Always )
            {
                is_open = g.NextItemData.OpenVal;
                storage->SetInt( id, is_open );
            }
            else
            {
                // We treat ImGuiCond_Once and ImGuiCond_FirstUseEver the same because tree node state are not saved persistently.
                const int stored_value = storage->GetInt( id, -1 );
                if( stored_value == -1 )
                {
                    is_open = g.NextItemData.OpenVal;
                    storage->SetInt( id, is_open );
                }
                else
                {
                    is_open = stored_value != 0;
                }
            }
        }
        else
        {
            is_open = storage->GetInt( id, ( flags & ImGuiTreeNodeFlags_DefaultOpen ) ? 1 : 0 ) != 0;
        }

        // When logging is enabled, we automatically expand tree nodes (but *NOT* collapsing headers.. seems like sensible behavior).
        // NB- If we are above max depth we still allow manually opened nodes to be logged.
        if( g.LogEnabled && !( flags & ImGuiTreeNodeFlags_NoAutoOpenOnLog ) &&
            ( window->DC.TreeDepth - g.LogDepthRef ) < g.LogDepthToExpand )
            is_open = true;

        return is_open;
    }

    void RenderArrow( ImDrawList *draw_list, ImVec2 pos, ImU32 col, ImGuiDir dir, float scale )
    {
        const float h      = draw_list->_Data->FontSize * 1.00f;
        float       r      = h * 0.45f * scale;
        ImVec2      center = pos + ImVec2( h * 0.50f, h * 0.50f * scale );

        ImVec2 a, b, c;
        switch( dir )
        {
        case ImGuiDir_Up:
        case ImGuiDir_Down:
            if( dir == ImGuiDir_Up ) r = -r;
            a = ImVec2( +0.000f, +0.750f ) * r;
            b = ImVec2( -0.866f, -0.750f ) * r;
            c = ImVec2( +0.866f, -0.750f ) * r;
            break;
        case ImGuiDir_Left:
        case ImGuiDir_Right:
            if( dir == ImGuiDir_Left ) r = -r;
            a = ImVec2( +0.750f, +0.000f ) * r;
            b = ImVec2( -0.750f, +0.866f ) * r;
            c = ImVec2( -0.750f, -0.866f ) * r;
            break;
        case ImGuiDir_None:
        case ImGuiDir_COUNT:
            IM_ASSERT( 0 );
            break;
        }
        draw_list->AddTriangleFilled( center + a, center + b, center + c, col );
    }

    bool TreeNodeBehavior( ImGuiID id, ImGuiTreeNodeFlags flags, const char *label, const char *label_end )
    {
        ImGuiWindow *window = ImGui::GetCurrentWindow();
        if( window->SkipItems ) return false;

        ImGuiContext     &g             = *GImGui;
        const ImGuiStyle &style         = g.Style;
        const bool        display_frame = ( flags & ImGuiTreeNodeFlags_Framed ) != 0;
        const ImVec2      padding       = ( display_frame || ( flags & ImGuiTreeNodeFlags_FramePadding ) )
                                              ? style.FramePadding
                                              : ImVec2( style.FramePadding.x, ImMin( window->DC.CurrLineTextBaseOffset, style.FramePadding.y ) );

        if( !label_end ) label_end = ImGui::FindRenderedTextEnd( label );
        const ImVec2 label_size = ImGui::CalcTextSize( label, label_end, false );

        // We vertically grow up to current line height up the typical widget height.
        const float frame_height =
            ImMax( ImMin( window->DC.CurrLineSize.y, g.FontSize + style.FramePadding.y * 2 ), label_size.y + padding.y * 2 );
        ImRect frame_bb;
        frame_bb.Min.x = ( flags & ImGuiTreeNodeFlags_SpanFullWidth ) ? window->WorkRect.Min.x : window->DC.CursorPos.x;
        frame_bb.Min.y = window->DC.CursorPos.y;
        frame_bb.Max.x = window->WorkRect.Max.x;
        frame_bb.Max.y = window->DC.CursorPos.y + frame_height;
        if( display_frame )
        {
            // Framed header expand a little outside the default padding, to the edge of InnerClipRect
            // (FIXME: May remove this at some point and make InnerClipRect align with WindowPadding.x instead of WindowPadding.x*0.5f)
            frame_bb.Min.x -= IM_FLOOR( window->WindowPadding.x * 0.5f - 1.0f );
            frame_bb.Max.x += IM_FLOOR( window->WindowPadding.x * 0.5f );
        }

        const float text_offset_x = g.FontSize + ( display_frame ? padding.x * 3 : padding.x * 2 ); // Collapser arrow width + Spacing
        const float text_offset_y =
            ImMax( padding.y, window->DC.CurrLineTextBaseOffset ); // Latch before ImGui::CalcTextSize changes it
        const float text_width = g.FontSize + ( label_size.x > 0.0f ? label_size.x + padding.x * 2 : 0.0f ); // Include collapser
        ImVec2      text_pos( window->DC.CursorPos.x + text_offset_x, window->DC.CursorPos.y + text_offset_y );
        ImGui::ItemSize( ImVec2( text_width, frame_height ), padding.y );

        // For regular tree nodes, we arbitrary allow to click past 2 worth of ItemSpacing
        ImRect interact_bb = frame_bb;
        if( !display_frame && ( flags & ( ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_SpanFullWidth ) ) == 0 )
            interact_bb.Max.x = frame_bb.Min.x + text_width + style.ItemSpacing.x * 2.0f;

        // Store a flag for the current depth to tell if we will allow closing this node when navigating one of its child.
        // For this purpose we essentially compare if g.NavIdIsAlive went from 0 to 1 between TreeNode() and TreePop().
        // This is currently only support 32 level deep and we are fine with (1 << Depth) overflowing into a zero.
        const bool is_leaf = ( flags & ImGuiTreeNodeFlags_Leaf ) != 0;
        bool       is_open = TreeNodeBehaviorIsOpen( id, flags );
        if( is_open && !g.NavIdIsAlive && ( flags & ImGuiTreeNodeFlags_NavLeftJumpsBackHere ) &&
            !( flags & ImGuiTreeNodeFlags_NoTreePushOnOpen ) )
            window->DC.TreeJumpToParentOnPopMask |= ( 1 << window->DC.TreeDepth );

        bool item_add = ImGui::ItemAdd( interact_bb, id );
        g.LastItemData.StatusFlags |= ImGuiItemStatusFlags_HasDisplayRect;
        g.LastItemData.DisplayRect = frame_bb;

        if( !item_add )
        {
            if( is_open && !( flags & ImGuiTreeNodeFlags_NoTreePushOnOpen ) ) TreePushOverrideID( id );
            IMGUI_TEST_ENGINE_ITEM_INFO( g.LastItemData.ID, label,
                g.LastItemData.StatusFlags | ( is_leaf ? 0 : ImGuiItemStatusFlags_Openable ) |
                    ( is_open ? ImGuiItemStatusFlags_Opened : 0 ) );
            return is_open;
        }

        ImGuiButtonFlags button_flags = ImGuiTreeNodeFlags_None;
        if( flags & ImGuiTreeNodeFlags_AllowItemOverlap ) button_flags |= ImGuiButtonFlags_AllowItemOverlap;
        if( !is_leaf ) button_flags |= ImGuiButtonFlags_PressedOnDragDropHold;

        // We allow clicking on the arrow section with keyboard modifiers held, in order to easily
        // allow browsing a tree while preserving selection with code implementing multi-selection patterns.
        // When clicking on the rest of the tree node we always disallow keyboard modifiers.
        const float arrow_hit_x1 = ( text_pos.x - text_offset_x ) - style.TouchExtraPadding.x;
        const float arrow_hit_x2 = ( text_pos.x - text_offset_x ) + ( g.FontSize + padding.x * 2.0f ) + style.TouchExtraPadding.x;
        const bool  is_mouse_x_over_arrow = ( g.IO.MousePos.x >= arrow_hit_x1 && g.IO.MousePos.x < arrow_hit_x2 );
        if( window != g.HoveredWindow || !is_mouse_x_over_arrow ) button_flags |= ImGuiButtonFlags_NoKeyModifiers;

        // Open behaviors can be altered with the _OpenOnArrow and _OnOnDoubleClick flags.
        // Some alteration have subtle effects (e.g. toggle on MouseUp vs MouseDown events) due to requirements for multi-selection and
        // drag and drop support.
        // - Single-click on label = Toggle on MouseUp (default, when _OpenOnArrow=0)
        // - Single-click on arrow = Toggle on MouseDown (when _OpenOnArrow=0)
        // - Single-click on arrow = Toggle on MouseDown (when _OpenOnArrow=1)
        // - Double-click on label = Toggle on MouseDoubleClick (when _OpenOnDoubleClick=1)
        // - Double-click on arrow = Toggle on MouseDoubleClick (when _OpenOnDoubleClick=1 and _OpenOnArrow=0)
        // It is rather standard that arrow click react on Down rather than Up.
        // We set ImGuiButtonFlags_PressedOnClickRelease on OpenOnDoubleClick because we want the item to be active on the initial
        // MouseDown in order for drag and drop to work.
        if( is_mouse_x_over_arrow )
            button_flags |= ImGuiButtonFlags_PressedOnClick;
        else if( flags & ImGuiTreeNodeFlags_OpenOnDoubleClick )
            button_flags |= ImGuiButtonFlags_PressedOnClickRelease | ImGuiButtonFlags_PressedOnDoubleClick;
        else
            button_flags |= ImGuiButtonFlags_PressedOnClickRelease;

        bool       selected     = ( flags & ImGuiTreeNodeFlags_Selected ) != 0;
        const bool was_selected = selected;

        bool hovered, held;
        bool pressed = ImGui::ButtonBehavior( interact_bb, id, &hovered, &held, button_flags );
        bool toggled = false;
        if( !is_leaf )
        {
            if( pressed && g.DragDropHoldJustPressedId != id )
            {
                if( ( flags & ( ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick ) ) == 0 ||
                    ( g.NavActivateId == id ) )
                    toggled = true;
                if( flags & ImGuiTreeNodeFlags_OpenOnArrow )
                    toggled |= is_mouse_x_over_arrow && !g.NavDisableMouseHover; // Lightweight equivalent of IsMouseHoveringRect()
                                                                                 // since ImGui::ButtonBehavior() already did the job
                if( ( flags & ImGuiTreeNodeFlags_OpenOnDoubleClick ) && g.IO.MouseClickedCount[0] == 2 ) toggled = true;
            }
            else if( pressed && g.DragDropHoldJustPressedId == id )
            {
                IM_ASSERT( button_flags & ImGuiButtonFlags_PressedOnDragDropHold );
                if( !is_open ) // When using Drag and Drop "hold to open" we keep the node highlighted after opening, but never close
                               // it again.
                    toggled = true;
            }

            if( g.NavId == id && g.NavMoveDir == ImGuiDir_Left && is_open )
            {
                toggled = true;
                ImGui::NavMoveRequestCancel();
            }
            if( g.NavId == id && g.NavMoveDir == ImGuiDir_Right &&
                !is_open ) // If there's something upcoming on the line we may want to give it the priority?
            {
                toggled = true;
                ImGui::NavMoveRequestCancel();
            }

            if( toggled )
            {
                is_open = !is_open;
                window->DC.StateStorage->SetInt( id, is_open );
                g.LastItemData.StatusFlags |= ImGuiItemStatusFlags_ToggledOpen;
            }
        }
        if( flags & ImGuiTreeNodeFlags_AllowItemOverlap ) ImGui::SetItemAllowOverlap();

        // In this branch, TreeNodeBehavior() cannot toggle the selection so this will never trigger.
        if( selected != was_selected ) //-V547
            g.LastItemData.StatusFlags |= ImGuiItemStatusFlags_ToggledSelection;

        // Render
        const ImU32            text_col            = ImGui::GetColorU32( ImGuiCol_Text );
        const ImU32            arrow_color         = IM_COL32(60, 60, 60, 100);
        ImGuiNavHighlightFlags nav_highlight_flags = ImGuiNavHighlightFlags_TypeThin;
        if( display_frame )
        {
            // Framed type
            const ImU32 bg_col = ImGui::GetColorU32( ( held && hovered ) ? ImGuiCol_HeaderActive
                                                     : hovered           ? ImGuiCol_HeaderHovered
                                                                         : ImGuiCol_Header );
            ImGui::RenderFrame( frame_bb.Min, frame_bb.Max, bg_col, true, style.FrameRounding );
            ImGui::RenderNavHighlight( frame_bb, id, nav_highlight_flags );
            if( flags & ImGuiTreeNodeFlags_Bullet )
                ImGui::RenderBullet(
                    window->DrawList, ImVec2( text_pos.x - text_offset_x * 0.60f, text_pos.y + g.FontSize * 0.5f ), arrow_color );
            else if( !is_leaf )
                RenderArrow( window->DrawList, ImVec2( text_pos.x - text_offset_x + padding.x, text_pos.y ), arrow_color,
                    is_open ? ImGuiDir_Down : ImGuiDir_Right, 1.0f );
            else // Leaf without bullet, left-adjusted text
                text_pos.x -= text_offset_x;
            if( flags & ImGuiTreeNodeFlags_ClipLabelForTrailingButton ) frame_bb.Max.x -= g.FontSize + style.FramePadding.x;

            if( g.LogEnabled ) ImGui::LogSetNextTextDecoration( "###", "###" );
            ImGui::RenderTextClipped( text_pos, frame_bb.Max, label, label_end, &label_size );
        }
        else
        {
            // Unframed typed for tree nodes
            if( hovered || selected )
            {
                const ImU32 bg_col = ImGui::GetColorU32( ( held && hovered ) ? ImGuiCol_HeaderActive
                                                         : hovered           ? ImGuiCol_HeaderHovered
                                                                             : ImGuiCol_Header );
                ImGui::RenderFrame( frame_bb.Min, frame_bb.Max, bg_col, false );
            }
            ImGui::RenderNavHighlight( frame_bb, id, nav_highlight_flags );
            if( flags & ImGuiTreeNodeFlags_Bullet )
                ImGui::RenderBullet(
                    window->DrawList, ImVec2( text_pos.x - text_offset_x * 0.5f, text_pos.y + g.FontSize * 0.5f ), arrow_color );
            else if( !is_leaf )
                RenderArrow( window->DrawList,
                    ImVec2( text_pos.x - text_offset_x + padding.x, text_pos.y + g.FontSize * 0.0f ), arrow_color,
                    is_open ? ImGuiDir_Down : ImGuiDir_Right, 1.0f );
            if( g.LogEnabled ) ImGui::LogSetNextTextDecoration( ">", NULL );
            ImGui::RenderText( text_pos, label, label_end, false );
        }

        if( is_open && !( flags & ImGuiTreeNodeFlags_NoTreePushOnOpen ) ) TreePushOverrideID( id );
        IMGUI_TEST_ENGINE_ITEM_INFO( id, label,
            g.LastItemData.StatusFlags | ( is_leaf ? 0 : ImGuiItemStatusFlags_Openable ) |
                ( is_open ? ImGuiItemStatusFlags_Opened : 0 ) );
        return is_open;
    }

    void TreePushOverrideID( ImGuiID id )
    {
        ImGuiContext &g      = *GImGui;
        ImGuiWindow  *window = g.CurrentWindow;
        ImGui::Indent( g.FontSize / 1.5f );
        window->DC.TreeDepth++;
        ImGui::PushOverrideID( id );
    }

    void TreePop()
    {
        ImGuiContext &g      = *GImGui;
        ImGuiWindow  *window = g.CurrentWindow;
        ImGui::Unindent( g.FontSize / 1.5f );

        window->DC.TreeDepth--;
        ImU32 tree_depth_mask = ( 1 << window->DC.TreeDepth );

        // Handle Left arrow to move to parent tree node (when ImGuiTreeNodeFlags_NavLeftJumpsBackHere is enabled)
        if( g.NavMoveDir == ImGuiDir_Left && g.NavWindow == window && ImGui::NavMoveRequestButNoResultYet() )
            if( g.NavIdIsAlive && ( window->DC.TreeJumpToParentOnPopMask & tree_depth_mask ) )
            {
                ImGui::SetNavID( window->IDStack.back(), g.NavLayer, 0, ImRect() );
                ImGui::NavMoveRequestCancel();
            }
        window->DC.TreeJumpToParentOnPopMask &= tree_depth_mask - 1;

        IM_ASSERT( window->IDStack.Size > 1 ); // There should always be 1 element in the IDStack (pushed during window creation). If
                                               // this triggers you called TreePop/ImGui::PopID too much.
        ImGui::PopID();
    }
} // namespace LTSE::Core::UI