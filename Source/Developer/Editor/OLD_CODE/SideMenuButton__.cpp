#include "SideMenuButton.h"

#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

namespace LTSE::Editor
{

    using namespace LTSE::Core;

    void SideMenuButton::Display( int32_t width, int32_t height )
    {
        auto l_DrawList = ImGui::GetWindowDrawList();

        ImVec2 l_TopLeft     = ImGui::GetCursorScreenPos();
        ImVec2 l_BottomRight = l_TopLeft + ImVec2{ 5.0f, static_cast<float>( width ) };

        Clicked = UI::Button( Title.c_str(), { width, width } );
        Hovered = ImGui::IsItemHovered();

        if( Selected )
        {
            l_DrawList->AddRectFilled( l_TopLeft, l_BottomRight, IM_COL32( 255, 255, 255, 200 ) );
        }
    }

} // namespace LTSE::Editor