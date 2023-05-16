#include "PopupWindow.h"

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

PopupWindow::PopupWindow( std::string a_Title, math::vec2 a_Size )
    : Title{ a_Title }
    , Size{ a_Size }
{
}

void PopupWindow::Display()
{
    ImGui::SetNextWindowSize( ImVec2{ Size.x, Size.y } );
    ImGuiWindowFlags l_Flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoResize |
                               ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2{ 15, 15 } );
    if( ImGui::BeginPopupModal( Title.c_str(), nullptr, l_Flags ) )
    {
        ImVec2 l_PopupSize = ImGui::GetWindowSize();

        WindowContent();

        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
}
