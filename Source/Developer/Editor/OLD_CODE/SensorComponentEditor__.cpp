#include "SensorComponentEditor.h"

#include "Developer/UI/CanvasView.h"
#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

#include "Developer/Scene/Components.h"

using namespace LTSE::Core::EntityComponentSystem::Components;

namespace LTSE::Editor
{

    SensorComponentEditor::SensorComponentEditor() { PhotodetectorEditor = PhotodetectorCellEditor( "EDIT PHOTODETECTOR...", { 1600.0f, 1200.0f } ); }

    void SensorComponentEditor::Display( int32_t width, int32_t height )
    {
        if( ComponentToEdit.Has<sPhotoDetector>() )
        {
            Title = "PHOTODETECTOR COMPONENT";
        }
        else if( ComponentToEdit.Has<sLaserAssembly>() )
        {
            Title = "LASER COMPONENT";
        }
        else if( ComponentToEdit.Has<sSampler>() )
        {
            Title = "SAMPLER COMPONENT";
        }
        else
        {
            Title = "UNKNOWN COMPONENT";
        }

        PropertiesPanel::Display( width, height );

        if( ComponentToEdit.Has<sPhotoDetector>() )
        {
            auto l_WindowSize = UI::GetAvailableContentSpace();
            auto l_TextSize0  = ImGui::CalcTextSize( "Cell positions:" );

            auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "ID:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f ) );
            UI::Text( "XXXX" );

            auto l_TextSize2 = ImGui::CalcTextSize( "Name:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "Name:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize2.x ) + 10.0f, 0.0f ) );
            char buf[128] = { 0 };
            std::strncpy( buf, ComponentToEdit.Get<sTag>().mValue.c_str(), std::min( ComponentToEdit.Get<sTag>().mValue.size(), std::size_t( 128 ) ) );
            if( ImGui::InputText( "##TAG_INPUT", buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
            {
                ComponentToEdit.Get<sTag>().mValue = std::string( buf );
            }

            auto l_TextSize3 = ImGui::CalcTextSize( "Cell positions:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "Cell positions:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize3.x ) + 10.0f, 0.0f ) );
            UI::Text( "XXXX" );

            auto l_TextSize4 = ImGui::CalcTextSize( "Baseline:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "Baseline:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize4.x ) + 10.0f, 0.0f ) );
            UI::Text( "XXXX" );

            auto l_TextSize5 = ImGui::CalcTextSize( "Static Noise:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "Static Noise:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize5.x ) + 10.0f, 0.0f ) );
            UI::Text( "XXXX" );

            if( UI::Button( "Edit layout", { 100.0f, 35.0f } ) )
            {
                PhotodetectorEditor.PhotodetectorToEdit = ComponentToEdit;
                PhotodetectorEditor.SensorModel         = SensorModel;
                PhotodetectorEditor.Visible             = true;
            }

            if( PhotodetectorEditor.Visible )
                ImGui::OpenPopup( "EDIT PHOTODETECTOR..." );
            PhotodetectorEditor.Display();
        }
        else if( ComponentToEdit.Has<sLaserAssembly>() )
        {
            auto l_WindowSize = UI::GetAvailableContentSpace();
            auto l_TextSize0  = ImGui::CalcTextSize( "Timebase delay:" );

            auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "ID:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f ) );
            UI::Text( "XXXX" );

            auto l_TextSize2 = ImGui::CalcTextSize( "Name:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "Name:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize2.x ) + 10.0f, 0.0f ) );
            char buf[128] = { 0 };
            std::strncpy( buf, ComponentToEdit.Get<sTag>().mValue.c_str(), std::min( ComponentToEdit.Get<sTag>().mValue.size(), std::size_t( 128 ) ) );
            if( ImGui::InputText( "##TAG_INPUT", buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
            {
                ComponentToEdit.Get<sTag>().mValue = std::string( buf );
            }

            auto l_TextSize3 = ImGui::CalcTextSize( "Pulse template:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "Pulse template:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize3.x ) + 10.0f, 0.0f ) );
            UI::Text( "XXXX" );

            auto l_TextSize4 = ImGui::CalcTextSize( "Timebase delay:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::VectorComponentEditor( "Timebase delay:", ComponentToEdit.Get<sLaserAssembly>().mTimebaseDelay, 0.0, l_TextSize0.x );

            auto l_TextSize5 = ImGui::CalcTextSize( "Flash time:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::VectorComponentEditor( "Flash time:", ComponentToEdit.Get<sLaserAssembly>().mFlashTime, 0.0, l_TextSize0.x );
        }
        else if( ComponentToEdit.Has<sSampler>() )
        {
            auto l_WindowSize = UI::GetAvailableContentSpace();
            auto l_TextSize0  = ImGui::CalcTextSize( "Timebase delay:" );

            auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "ID:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f ) );
            UI::Text( "XXXX" );

            auto l_TextSize2 = ImGui::CalcTextSize( "Name:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "Name:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize2.x ) + 10.0f, 0.0f ) );
            char buf[128] = { 0 };
            std::strncpy( buf, ComponentToEdit.Get<sTag>().mValue.c_str(), std::min( ComponentToEdit.Get<sTag>().mValue.size(), std::size_t( 128 ) ) );
            if( ImGui::InputText( "##TAG_INPUT", buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
            {
                ComponentToEdit.Get<sTag>().mValue = std::string( buf );
            }

            auto l_TextSize3 = ImGui::CalcTextSize( "Basepoints:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "Basepoints:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize3.x ) + 10.0f, 0.0f ) );
            UI::Text( "{}", ComponentToEdit.Get<sSampler>().mLength );

            auto l_TextSize4 = ImGui::CalcTextSize( "Frequency:" );
            UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
            UI::Text( "Frequency:" );
            UI::SameLine();
            UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize4.x ) + 10.0f, 0.0f ) );
            UI::Text( "{}", ComponentToEdit.Get<sSampler>().mFrequency );
        }
        else
        {
        }
    }

} // namespace LTSE::Editor