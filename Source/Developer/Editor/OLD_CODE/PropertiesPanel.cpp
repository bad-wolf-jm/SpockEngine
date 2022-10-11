#include "PropertiesPanel.h"

#include "Developer/UI/UI.h"

namespace LTSE::Editor
{
    using namespace LTSE::Core;

    void PropertiesPanel::Display( int32_t width, int32_t height )
    {
        auto l_DrawList   = ImGui::GetWindowDrawList();
        auto l_WindowSize = UI::GetAvailableContentSpace();
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        auto l_TopLeft0     = ImGui::GetCursorScreenPos() + ImVec2{ -10.0f, -10.0f };
        auto l_BottomRight0 = ImGui::GetCursorScreenPos() + ImVec2{ static_cast<float>( l_WindowSize.x ), 25.0f };
        l_DrawList->AddRectFilled( l_TopLeft0, l_BottomRight0, IM_COL32( 5, 5, 5, 255 ) );
        UI::Text( Title );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 0.0f, 15.0f ) );
    }

} // namespace LTSE::Editor