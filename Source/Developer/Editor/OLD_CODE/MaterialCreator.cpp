#include "MaterialCreator.h"
#include "Developer/UI/UI.h"

namespace LTSE::Editor
{

    MaterialCreator::MaterialCreator( std::string a_Title, math::vec2 a_Size )
        : PopupWindow( a_Title, a_Size )
    {
    }

    void MaterialCreator::WindowContent()
    {
        if( !Visible )
            return;
        ImVec2 l_PopupSize = ImGui::GetWindowSize();

        UI::SetCursorPosition( l_PopupSize - ImVec2{ 150, 40 } - ImVec2{ 15, 15 } );
        if( ImGui::Button( "Cancel##MaterialCreator", ImVec2{ 150, 40 } ) )
        {
            ImGui::CloseCurrentPopup();
            Visible = false;
        }
    }

} // namespace LTSE::Editor