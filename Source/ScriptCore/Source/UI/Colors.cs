using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public enum UIStyleColor
    {
        Text,
        TextDisabled,
        WindowBg,              // Background of normal windows
        ChildBg,               // Background of child windows
        PopupBg,               // Background of popups, menus, tooltips windows
        Border,
        BorderShadow,
        FrameBg,               // Background of checkbox, radio button, plot, slider, text input
        FrameBgHovered,
        FrameBgActive,
        TitleBg,
        TitleBgActive,
        TitleBgCollapsed,
        MenuBarBg,
        ScrollbarBg,
        ScrollbarGrab,
        ScrollbarGrabHovered,
        ScrollbarGrabActive,
        CheckMark,
        SliderGrab,
        SliderGrabActive,
        Button,
        ButtonHovered,
        ButtonActive,
        Header,                // Header* colors are used for CollapsingHeader, TreeNode, Selectable, MenuItem
        HeaderHovered,
        HeaderActive,
        Separator,
        SeparatorHovered,
        SeparatorActive,
        ResizeGrip,
        ResizeGripHovered,
        ResizeGripActive,
        Tab,
        TabHovered,
        TabActive,
        TabUnfocused,
        TabUnfocusedActive,
        DockingPreview,        // Preview overlay color when about to docking something
        DockingEmptyBg,        // Background color for empty node (e.g. CentralNode with no window docked into it)
        PlotLines,
        PlotLinesHovered,
        PlotHistogram,
        PlotHistogramHovered,
        TableHeaderBg,         // Table header background
        TableBorderStrong,     // Table outer and header borders (prefer using Alpha=1.0 here)
        TableBorderLight,      // Table inner borders (prefer using Alpha=1.0 here)
        TableRowBg,            // Table row background (even rows)
        TableRowBgAlt,         // Table row background (odd rows)
        TextSelectedBg,
        DragDropTarget,
        NavHighlight,          // Gamepad/keyboard: current highlighted item
        NavWindowingHighlight, // Highlight window when using CTRL+TAB
        NavWindowingDimBg,     // Darken/colorize entire screen behind the CTRL+TAB window list, when active
        ModalWindowDimBg       // Darken/colorize entire screen behind a modal window, when one is active
    };

    public static class UIColor
    {
        public static Math.vec4 Color(byte aRed, byte aGreen, byte aBlue, byte aAlpha)
        {
            return new Math.vec4((float)aRed / 255.0f, (float)aGreen / 255.0f, (float)aBlue / 255.0f, (float)aAlpha / 255.0f);
        }

        public static Math.vec4 Color(byte aRed, byte aGreen, byte aBlue, float aAlpha)
        {
            return new Math.vec4((float)aRed / 255.0f, (float)aGreen / 255.0f, (float)aBlue / 255.0f, aAlpha);
        }

        public static Math.vec4 Color(float aRed, float aGreen, float aBlue, float aAlpha)
        {
            return new Math.vec4(aRed, aGreen, aBlue, aAlpha);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        public extern static Math.vec4 GetStyleColor(UIStyleColor aColor);
    }

}