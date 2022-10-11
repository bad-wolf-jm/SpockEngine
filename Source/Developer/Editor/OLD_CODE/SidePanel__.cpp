#include "SidePanel.h"

#include "Developer/UI/UI.h"

namespace LTSE::Editor
{

    using namespace LTSE::Core;

    void SidePanel::Display( int32_t width, int32_t height )
    {
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 15.0f, 15.0f ) );
        UI::Text( Title );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 0.0f, 20.0f ) );
    }

} // namespace LTSE::Editor