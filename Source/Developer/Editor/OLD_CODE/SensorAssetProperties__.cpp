#include "SensorAssetProperties.h"

#include "Developer/UI/CanvasView.h"
#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

#include "Developer/Scene/Components.h"

using namespace LTSE::Core::EntityComponentSystem::Components;

namespace LTSE::Editor
{

    SensorAssetProperties::SensorAssetProperties() {}

    void SensorAssetProperties::Display( int32_t width, int32_t height )
    {
        Title = "ASSET METADATA";

        PropertiesPanel::Display( width, height );

        auto l_WindowSize = UI::GetAvailableContentSpace();

        auto l_TextSize0  = ImGui::CalcTextSize( "Cell positions:" );
        auto l_TextSize1 = ImGui::CalcTextSize( "ID:" );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        UI::Text( "ID:" );
        UI::SameLine();
        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize1.x ) + 10.0f, 0.0f ) );
        UI::Text( ComponentToEdit.Get<sAssetMetadata>().mID );

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

        auto l_TextSize3 = ImGui::CalcTextSize( "Type:" );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        UI::Text( "Type:" );
        UI::SameLine();
        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize3.x ) + 10.0f, 0.0f ) );
        if( ComponentToEdit.Has<sDiffusionAssetTag>() )
        {
            UI::Text("DIFFUSION");
        }
        else if( ComponentToEdit.Has<sReductionMapAssetTag>() )
        {
            UI::Text("REDUCTION");
        }
        else if( ComponentToEdit.Has<sPulseTemplateAssetTag>() )
        {
            UI::Text("PULSE TEMPLATE");
        }
        else if( ComponentToEdit.Has<sStaticNoiseAssetTag>() )
        {
            UI::Text("STATIC_NOISE");
        }
        else
        {
            UI::Text("UNKNOWN ASSET");
        }

        auto l_TextSize4 = ImGui::CalcTextSize( "Asset root:" );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        UI::Text( "Asset root:" );
        UI::SameLine();
        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize4.x ) + 10.0f, 0.0f ) );
        UI::Text( ComponentToEdit.Get<sAssetLocation>().mRoot.string() );

        auto l_TextSize5 = ImGui::CalcTextSize( "Asset path:" );
        UI::SetCursorPosition( UI::GetCurrentCursorPosition() + math::vec2( 10.0f, 10.0f ) );
        UI::Text( "Asset path:" );
        UI::SameLine();
        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( ( l_TextSize0.x - l_TextSize5.x ) + 10.0f, 0.0f ) );
        UI::Text( ComponentToEdit.Get<sAssetLocation>().mFilePath.string() );

        if( ComponentToEdit.Has<sDiffusionAssetTag>() )
        {
        }
        else if( ComponentToEdit.Has<sReductionMapAssetTag>() )
        {
        }
        else if( ComponentToEdit.Has<sPulseTemplateAssetTag>() )
        {
        }
        else if( ComponentToEdit.Has<sStaticNoiseAssetTag>() )
        {
        }
        else
        {
        }
    }

} // namespace LTSE::Editor