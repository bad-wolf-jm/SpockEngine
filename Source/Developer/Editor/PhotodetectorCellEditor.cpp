#include "PhotodetectorCellEditor.h"

#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

#include "LidarSensorModel/Components.h"
#include <fmt/core.h>

using namespace LTSE::Core::UI;
using namespace LTSE::SensorModel;

namespace LTSE::Editor
{

    PhotodetectorCellEditor::PhotodetectorCellEditor( std::string a_Title, math::vec2 a_Size )
        : PopupWindow( a_Title, a_Size )
    {
        Canvas.XAxisBounds = { -180.0f, 180.0f };
        Canvas.YAxisBounds = { -180.0f, 180.0f };
        Canvas.MajorUnit   = 10.0f;
        Canvas.MinorUnit   = 10.0f;
    }

    void PhotodetectorCellEditor::WindowContent()
    {
        if( !Visible )
            return;

        ImVec2 l_PopupSize = ImGui::GetWindowSize();

        Canvas.Display( [&]( auto *x ) {} );

        UI::SetCursorPosition( l_PopupSize - ImVec2{ 150, 40 } - ImVec2{ 15, 15 } );
        if( ImGui::Button( "Cancel##PhotodetectorCellEditor", ImVec2{ 150, 40 } ) )
        {
            ImGui::CloseCurrentPopup();
            Visible             = false;
            PhotodetectorToEdit = Entity{};
        }
    }

} // namespace LTSE::Editor