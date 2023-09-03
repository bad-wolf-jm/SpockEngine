#include "UI.h"

namespace SE::Core::UI
{

    UIStyle::UIStyle( [[maybe_unused]] bool x )
    {
        // UNUSED( x );

        ImGuiStyle &style  = ImGui::GetStyle();
        ImVec4     *colors = style.Colors;

        // Background colors
        colors[ImGuiCol_WindowBg]         = ImVec4( 0.01f, 0.01f, 0.01f, 1.0f );
        colors[ImGuiCol_ChildBg]          = ImVec4( 0.01f, 0.01f, 0.01f, 0.0f );
        colors[ImGuiCol_PopupBg]          = ImVec4( 0.01f, 0.01f, 0.01f, 1.0f );
        colors[ImGuiCol_MenuBarBg]        = ImVec4( 0.035f, 0.035f, 0.035f, 0.0f );
        colors[ImGuiCol_FrameBg]          = ImVec4( 0.42f, 0.42f, 0.42f, 0.0f );
        colors[ImGuiCol_FrameBgHovered]   = ImVec4( 0.42f, 0.42f, 0.42f, 0.0f );
        colors[ImGuiCol_FrameBgActive]    = ImVec4( 0.42f, 0.42f, 0.42f, 0.0f );
        colors[ImGuiCol_ScrollbarBg]      = ImVec4( 0.24f, 0.24f, 0.24f, 0.0f );
        colors[ImGuiCol_TitleBg]          = ImVec4( 0.015f, 0.015f, 0.015f, 1.0f );
        colors[ImGuiCol_TitleBgActive]    = ImVec4( 0.025f, 0.025f, 0.025f, 1.0f );
        colors[ImGuiCol_TitleBgCollapsed] = ImVec4( 0.015f, 0.015f, 0.015f, 1.0f );
        colors[ImGuiCol_Header]           = ImVec4( 0.01f, 0.01f, 0.01f, 0.0f );
        colors[ImGuiCol_HeaderHovered]    = ImVec4( 0.02f, 0.02f, 0.02f, 0.0f );
        colors[ImGuiCol_HeaderActive]     = ImVec4( 0.02f, 0.02f, 0.02f, 0.0f );

        colors[ImGuiCol_Tab]                = ImVec4( 0.02f, 0.02f, 0.02f, 1.0f );
        colors[ImGuiCol_TabHovered]         = ImVec4( 0.05f, 0.05f, 0.05f, 1.0f );
        colors[ImGuiCol_TabActive]          = ImVec4( 0.06f, 0.06f, 0.06f, 1.0f );
        colors[ImGuiCol_TabUnfocused]       = ImVec4( 0.06f, 0.06f, 0.06f, 1.0f );
        colors[ImGuiCol_TabUnfocusedActive] = ImVec4( 0.03f, 0.03f, 0.03f, 1.0f );

        colors[ImGuiCol_TableBorderStrong] = ImVec4( .02f, 0.02f, 0.02f, 1.0f );
        colors[ImGuiCol_TableBorderLight]  = ImVec4( .045f, 0.045f, 0.045f, 1.0f );
        colors[ImGuiCol_TableHeaderBg]     = ImVec4( 0.02f, 0.02f, 0.02f, 0.0f );
        colors[ImGuiCol_TableRowBg]        = ImVec4( 0.0f, 0.0f, 0.0f, 0.0f );
        colors[ImGuiCol_TableRowBgAlt]     = ImVec4( 0.02f, 0.02f, 0.02f, 0.0f );
        colors[ImGuiCol_DockingEmptyBg]    = ImVec4( 0.0f, 0.0f, 0.0f, 0.0f );

        colors[ImGuiCol_Text]           = ImVec4( .6f, .6f, .6f, 1.0f );
        colors[ImGuiCol_TextDisabled]   = ImVec4( 0.1f, 0.1f, 0.1f, 1.0f );
        colors[ImGuiCol_TextSelectedBg] = ImVec4( 0.73f, 0.73f, 0.73f, 0.35f );

        colors[ImGuiCol_Border]       = ImVec4( 0.29f, 0.29f, 0.29f, 1.0f );
        colors[ImGuiCol_BorderShadow] = ImVec4( 1.0f, 1.0f, 1.0f, 0.06f );

        colors[ImGuiCol_ScrollbarGrab]        = ImVec4( 0.41f, 0.41f, 0.41f, 1.0f );
        colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4( 0.52f, 0.52f, 0.52f, 1.0f );
        colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4( 0.76f, 0.76f, 0.76f, 1.0f );
        colors[ImGuiCol_CheckMark]            = ImVec4( 0.65f, 0.65f, 0.65f, 1.0f );
        colors[ImGuiCol_SliderGrab]           = ImVec4( 0.52f, 0.52f, 0.52f, 1.0f );
        colors[ImGuiCol_SliderGrabActive]     = ImVec4( 0.64f, 0.64f, 0.64f, 1.0f );

        colors[ImGuiCol_Button]        = ImVec4( 0.54f, 0.54f, 0.54f, 0.0f );
        colors[ImGuiCol_ButtonHovered] = ImVec4( 0.52f, 0.52f, 0.52f, 0.09f );
        colors[ImGuiCol_ButtonActive]  = ImVec4( 0.76f, 0.76f, 0.76f, 0.0f );

        colors[ImGuiCol_Separator]         = ImVec4( 0.05f, 0.05f, 0.05f, 1.f );
        colors[ImGuiCol_SeparatorHovered]  = ImVec4( 0.02f, 0.02f, 0.04f, 1.f );
        colors[ImGuiCol_SeparatorActive]   = ImVec4( 0.02f, 0.02f, 0.02f, 1.f );
        colors[ImGuiCol_ResizeGrip]        = ImVec4( 0.26f, 0.59f, 0.98f, 0.25f );
        colors[ImGuiCol_ResizeGripHovered] = ImVec4( 0.26f, 0.59f, 0.98f, 0.67f );
        colors[ImGuiCol_ResizeGripActive]  = ImVec4( 0.26f, 0.59f, 0.98f, 0.95f );

        colors[ImGuiCol_PlotLines]             = ImVec4( 0.61f, 0.61f, 0.61f, 1.0f );
        colors[ImGuiCol_PlotLinesHovered]      = ImVec4( 1.0f, 0.43f, 0.35f, 1.0f );
        colors[ImGuiCol_PlotHistogram]         = ImVec4( 0.90f, 0.70f, 0.00f, 1.0f );
        colors[ImGuiCol_PlotHistogramHovered]  = ImVec4( 1.0f, 0.60f, 0.00f, 1.0f );
        colors[ImGuiCol_ModalWindowDimBg]      = ImVec4( 0.00f, 0.00f, 0.00f, 0.95f );
        colors[ImGuiCol_DragDropTarget]        = ImVec4( 1.0f, 1.0f, 0.00f, 0.90f );
        colors[ImGuiCol_NavHighlight]          = ImVec4( 0.26f, 0.59f, 0.98f, 1.0f );
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4( 1.0f, 1.0f, 1.0f, 0.70f );
        colors[ImGuiCol_NavWindowingDimBg]     = ImVec4( 0.80f, 0.80f, 0.80f, 0.20f );

        style.PopupRounding     = 0;
        style.WindowPadding     = ImVec2( 4, 4 );
        style.FramePadding      = ImVec2( 6, 6 );
        style.ItemSpacing       = ImVec2( 6, 4 );
        style.ScrollbarSize     = 6;
        style.WindowBorderSize  = 1;
        style.ChildBorderSize   = 0;
        style.PopupBorderSize   = 1;
        style.FrameBorderSize   = 0;
        style.WindowRounding    = 0;
        style.ChildRounding     = 0;
        style.TabRounding       = 0;
        style.FrameRounding     = 0;
        style.ScrollbarRounding = 2;
        style.GrabRounding      = 3;
    }
} // namespace SE::Core::UI