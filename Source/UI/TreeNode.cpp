
namespace LTSE::Core::UI
{

    // bool TreeNode( const char *str_id, const char *fmt, ... )
    // {
    //     va_list args;
    //     va_start( args, fmt );
    //     bool is_open = TreeNodeExV( str_id, 0, fmt, args );
    //     va_end( args );
    //     return is_open;
    // }

    // bool TreeNode( const void *ptr_id, const char *fmt, ... )
    // {
    //     va_list args;
    //     va_start( args, fmt );
    //     bool is_open = TreeNodeExV( ptr_id, 0, fmt, args );
    //     va_end( args );
    //     return is_open;
    // }

    bool TreeNode( const char *label )
    {
        ImGuiWindow *window = GetCurrentWindow();
        if( window->SkipItems ) return false;
        return TreeNodeBehavior( window->GetID( label ), 0, label, NULL );
    }

    bool TreeNodeV( const char *str_id, const char *fmt, va_list args ) { return TreeNodeExV( str_id, 0, fmt, args ); }

    bool TreeNodeV( const void *ptr_id, const char *fmt, va_list args ) { return TreeNodeExV( ptr_id, 0, fmt, args ); }

    bool TreeNodeEx( const char *label, ImGuiTreeNodeFlags flags )
    {
        ImGuiWindow *window = GetCurrentWindow();
        if( window->SkipItems ) return false;

        return TreeNodeBehavior( window->GetID( label ), flags, label, NULL );
    }

    // bool TreeNodeEx( const char *str_id, ImGuiTreeNodeFlags flags, const char *fmt, ... )
    // {
    //     va_list args;
    //     va_start( args, fmt );
    //     bool is_open = TreeNodeExV( str_id, flags, fmt, args );
    //     va_end( args );
    //     return is_open;
    // }

    // bool TreeNodeEx( const void *ptr_id, ImGuiTreeNodeFlags flags, const char *fmt, ... )
    // {
    //     va_list args;
    //     va_start( args, fmt );
    //     bool is_open = TreeNodeExV( ptr_id, flags, fmt, args );
    //     va_end( args );
    //     return is_open;
    // }

    // bool TreeNodeExV( const char *str_id, ImGuiTreeNodeFlags flags, const char *fmt, va_list args )
    // {
    //     ImGuiWindow *window = GetCurrentWindow();
    //     if( window->SkipItems ) return false;

    //     ImGuiContext &g         = *GImGui;
    //     const char   *label_end = g.TempBuffer + ImFormatStringV( g.TempBuffer, IM_ARRAYSIZE( g.TempBuffer ), fmt, args );
    //     return TreeNodeBehavior( window->GetID( str_id ), flags, g.TempBuffer, label_end );
    // }

    // bool TreeNodeExV( const void *ptr_id, ImGuiTreeNodeFlags flags, const char *fmt, va_list args )
    // {
    //     ImGuiWindow *window = GetCurrentWindow();
    //     if( window->SkipItems ) return false;

    //     ImGuiContext &g         = *GImGui;
    //     const char   *label_end = g.TempBuffer + ImFormatStringV( g.TempBuffer, IM_ARRAYSIZE( g.TempBuffer ), fmt, args );
    //     return TreeNodeBehavior( window->GetID( ptr_id ), flags, g.TempBuffer, label_end );
    // }

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

    bool TreeNodeBehavior( ImGuiID id, ImGuiTreeNodeFlags flags, const char *label, const char *label_end )
    {
        ImGuiWindow *window = GetCurrentWindow();
        if( window->SkipItems ) return false;

        ImGuiContext     &g             = *GImGui;
        const ImGuiStyle &style         = g.Style;
        const bool        display_frame = ( flags & ImGuiTreeNodeFlags_Framed ) != 0;
        const ImVec2      padding       = ( display_frame || ( flags & ImGuiTreeNodeFlags_FramePadding ) )
                                              ? style.FramePadding
                                              : ImVec2( style.FramePadding.x, ImMin( window->DC.CurrLineTextBaseOffset, style.FramePadding.y ) );

        if( !label_end ) label_end = FindRenderedTextEnd( label );
        const ImVec2 label_size = CalcTextSize( label, label_end, false );

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
        const float text_offset_y = ImMax( padding.y, window->DC.CurrLineTextBaseOffset );          // Latch before ItemSize changes it
        const float text_width    = g.FontSize + ( label_size.x > 0.0f ? label_size.x + padding.x * 2 : 0.0f ); // Include collapser
        ImVec2      text_pos( window->DC.CursorPos.x + text_offset_x, window->DC.CursorPos.y + text_offset_y );
        ItemSize( ImVec2( text_width, frame_height ), padding.y );

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

        bool item_add = ItemAdd( interact_bb, id );
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
        bool pressed = ButtonBehavior( interact_bb, id, &hovered, &held, button_flags );
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
                                                                                 // since ButtonBehavior() already did the job
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
                NavMoveRequestCancel();
            }
            if( g.NavId == id && g.NavMoveDir == ImGuiDir_Right &&
                !is_open ) // If there's something upcoming on the line we may want to give it the priority?
            {
                toggled = true;
                NavMoveRequestCancel();
            }

            if( toggled )
            {
                is_open = !is_open;
                window->DC.StateStorage->SetInt( id, is_open );
                g.LastItemData.StatusFlags |= ImGuiItemStatusFlags_ToggledOpen;
            }
        }
        if( flags & ImGuiTreeNodeFlags_AllowItemOverlap ) SetItemAllowOverlap();

        // In this branch, TreeNodeBehavior() cannot toggle the selection so this will never trigger.
        if( selected != was_selected ) //-V547
            g.LastItemData.StatusFlags |= ImGuiItemStatusFlags_ToggledSelection;

        // Render
        const ImU32            text_col            = GetColorU32( ImGuiCol_Text );
        ImGuiNavHighlightFlags nav_highlight_flags = ImGuiNavHighlightFlags_TypeThin;
        if( display_frame )
        {
            // Framed type
            const ImU32 bg_col = GetColorU32( ( held && hovered ) ? ImGuiCol_HeaderActive
                                              : hovered           ? ImGuiCol_HeaderHovered
                                                                  : ImGuiCol_Header );
            RenderFrame( frame_bb.Min, frame_bb.Max, bg_col, true, style.FrameRounding );
            RenderNavHighlight( frame_bb, id, nav_highlight_flags );
            if( flags & ImGuiTreeNodeFlags_Bullet )
                RenderBullet(
                    window->DrawList, ImVec2( text_pos.x - text_offset_x * 0.60f, text_pos.y + g.FontSize * 0.5f ), text_col );
            else if( !is_leaf )
                RenderArrow( window->DrawList, ImVec2( text_pos.x - text_offset_x + padding.x, text_pos.y ), text_col,
                    is_open ? ImGuiDir_Down : ImGuiDir_Right, 1.0f );
            else // Leaf without bullet, left-adjusted text
                text_pos.x -= text_offset_x;
            if( flags & ImGuiTreeNodeFlags_ClipLabelForTrailingButton ) frame_bb.Max.x -= g.FontSize + style.FramePadding.x;

            if( g.LogEnabled ) LogSetNextTextDecoration( "###", "###" );
            RenderTextClipped( text_pos, frame_bb.Max, label, label_end, &label_size );
        }
        else
        {
            // Unframed typed for tree nodes
            if( hovered || selected )
            {
                const ImU32 bg_col = GetColorU32( ( held && hovered ) ? ImGuiCol_HeaderActive
                                                  : hovered           ? ImGuiCol_HeaderHovered
                                                                      : ImGuiCol_Header );
                RenderFrame( frame_bb.Min, frame_bb.Max, bg_col, false );
            }
            RenderNavHighlight( frame_bb, id, nav_highlight_flags );
            if( flags & ImGuiTreeNodeFlags_Bullet )
                RenderBullet(
                    window->DrawList, ImVec2( text_pos.x - text_offset_x * 0.5f, text_pos.y + g.FontSize * 0.5f ), text_col );
            else if( !is_leaf )
                RenderArrow( window->DrawList, ImVec2( text_pos.x - text_offset_x + padding.x, text_pos.y + g.FontSize * 0.15f ),
                    text_col, is_open ? ImGuiDir_Down : ImGuiDir_Right, 0.70f );
            if( g.LogEnabled ) LogSetNextTextDecoration( ">", NULL );
            RenderText( text_pos, label, label_end, false );
        }

        if( is_open && !( flags & ImGuiTreeNodeFlags_NoTreePushOnOpen ) ) TreePushOverrideID( id );
        IMGUI_TEST_ENGINE_ITEM_INFO( id, label,
            g.LastItemData.StatusFlags | ( is_leaf ? 0 : ImGuiItemStatusFlags_Openable ) |
                ( is_open ? ImGuiItemStatusFlags_Opened : 0 ) );
        return is_open;
    }

    void TreePush( const char *str_id )
    {
        ImGuiWindow *window = GetCurrentWindow();
        Indent();
        window->DC.TreeDepth++;
        PushID( str_id );
    }

    void TreePush( const void *ptr_id )
    {
        ImGuiWindow *window = GetCurrentWindow();
        Indent();
        window->DC.TreeDepth++;
        PushID( ptr_id );
    }

    void TreePushOverrideID( ImGuiID id )
    {
        ImGuiContext &g      = *GImGui;
        ImGuiWindow  *window = g.CurrentWindow;
        Indent();
        window->DC.TreeDepth++;
        PushOverrideID( id );
    }

    void TreePop()
    {
        ImGuiContext &g      = *GImGui;
        ImGuiWindow  *window = g.CurrentWindow;
        Unindent();

        window->DC.TreeDepth--;
        ImU32 tree_depth_mask = ( 1 << window->DC.TreeDepth );

        // Handle Left arrow to move to parent tree node (when ImGuiTreeNodeFlags_NavLeftJumpsBackHere is enabled)
        if( g.NavMoveDir == ImGuiDir_Left && g.NavWindow == window && NavMoveRequestButNoResultYet() )
            if( g.NavIdIsAlive && ( window->DC.TreeJumpToParentOnPopMask & tree_depth_mask ) )
            {
                SetNavID( window->IDStack.back(), g.NavLayer, 0, ImRect() );
                NavMoveRequestCancel();
            }
        window->DC.TreeJumpToParentOnPopMask &= tree_depth_mask - 1;

        IM_ASSERT( window->IDStack.Size > 1 ); // There should always be 1 element in the IDStack (pushed during window creation). If
                                               // this triggers you called TreePop/PopID too much.
        PopID();
    }

    // Horizontal distance preceding label when using TreeNode() or Bullet()
    float GetTreeNodeToLabelSpacing()
    {
        ImGuiContext &g = *GImGui;
        return g.FontSize + ( g.Style.FramePadding.x * 2.0f );
    }
} // namespace LTSE::Core::UI