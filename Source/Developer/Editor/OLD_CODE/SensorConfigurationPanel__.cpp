#include "SensorConfigurationPanel.h"

#include "Developer/UI/CanvasView.h"
#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

namespace LTSE::Editor
{

    static bool EditButton( Entity a_Node, math::vec2 a_Size )
    {
        char l_OnLabel[128];
        sprintf( l_OnLabel, "%s##%d", ICON_FA_PENCIL_SQUARE_O, (uint32_t)a_Node );

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0, 0.0, 0.0, 0.0 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0, 1.0, 1.0, 0.10 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0, 1.0, 1.0, 0.20 } );

        bool l_IsVisible;
        bool l_DoEdit =  UI::Button( l_OnLabel, a_Size );

        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        return l_DoEdit;
    }

    void SensorConfigurationPanel::Display( int32_t width, int32_t height )
    {
        auto l_DrawList   = ImGui::GetWindowDrawList();
        auto l_WindowSize = UI::GetAvailableContentSpace();

        if( !SensorModel )
            return;

        auto l_TopLeft     = ImGui::GetCursorScreenPos() + ImVec2{ -10.0f, -10.0f };
        auto l_BottomRight = ImGui::GetCursorScreenPos() + ImVec2{ static_cast<float>( l_WindowSize.x ), 25.0f };
        l_DrawList->AddRectFilled( l_TopLeft, l_BottomRight, IM_COL32( 5, 5, 5, 255 ) );
        if( !( SensorModel->mSensorDefinition->mName.empty() ) )
            UI::Text( "{}", SensorModel->mSensorDefinition->mName );
        else
            UI::Text( "SENSOR_001" );

        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 10.0f ) );
        ImGuiTreeNodeFlags l_Flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen;
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 2 ) );

        if( ImGui::TreeNodeEx( "ASSETS", l_Flags ) )
        {
            for( auto &l_Tile : SensorModel->mSensorDefinition->mRootAsset.Get<sRelationshipComponent>().mChildren )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
                UI::Text( "{}", ICON_FA_CIRCLE_O );
                UI::SameLine();
                ImGui::PopStyleColor();

                UI::Text( l_Tile.Get<sTag>().mValue );
                UI::SameLine();
                UI::SetCursorPosition( math::vec2( width - 30.0f, UI::GetCurrentCursorPosition().y - 3.0f ) );
                if( EditButton( l_Tile, math::vec2{ 20.0, 22.0 } ) )
                {
                    TileInFocus                     = l_Tile;
                    AssetProperties.ComponentToEdit = l_Tile;
                    AssetProperties.SensorModel     = SensorModel;
                    RequestPropertyEditor           = PropertyPanelID::SENSOR_ASSET_EDITOR;
                    ImGui::SetWindowFocus( "PROPERTIES" );
                }
            }
            ImGui::TreePop();
        }
        ImGui::PopStyleVar();

        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 15.0f ) );
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 2 ) );
        if( ImGui::TreeNodeEx( "COMPONENTS", l_Flags ) )
        {
            for( auto &l_Tile : SensorModel->mSensorDefinition->mRootComponent.Get<sRelationshipComponent>().mChildren )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
                UI::Text( "{}", ICON_FA_CIRCLE_O );
                UI::SameLine();
                ImGui::PopStyleColor();

                UI::Text( l_Tile.Get<sTag>().mValue );
                UI::SameLine();
                UI::SetCursorPosition( math::vec2( width - 30.0f, UI::GetCurrentCursorPosition().y - 3.0f ) );
                if( EditButton( l_Tile, math::vec2{ 20.0, 22.0 } ) )
                {
                    TileInFocus                     = l_Tile;
                    ComponentEditor.ComponentToEdit = l_Tile;
                    ComponentEditor.SensorModel     = SensorModel;
                    RequestPropertyEditor           = PropertyPanelID::SENSOR_COMPONENT_EDITOR;
                    ImGui::SetWindowFocus( "PROPERTIES" );
                }
            }
            ImGui::TreePop();
        }
        ImGui::PopStyleVar();

        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 15.0f ) );
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 2 ) );
        if( ImGui::TreeNodeEx( "TILE CONFIGURATIONS", l_Flags ) )
        {
            for( auto &l_Tile : SensorModel->mSensorDefinition->mRootTile.Get<sRelationshipComponent>().mChildren )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
                UI::Text( "{}", ICON_FA_CIRCLE_O );
                UI::SameLine();
                ImGui::PopStyleColor();
                UI::Text( "Tile ID: {} ({} flashes)", l_Tile.Get<sTileSpecificationComponent>().mID, l_Tile.Get<sRelationshipComponent>().mChildren.size() );
                UI::SameLine();
                UI::SetCursorPosition( math::vec2( width - 30.0f, UI::GetCurrentCursorPosition().y - 3.0f ) );
                if( EditButton( l_Tile, math::vec2{ 20.0, 22.0 } ) )
                {
                    TileInFocus               = l_Tile;
                    ElementEditor.TileToEdit  = l_Tile;
                    ElementEditor.SensorModel = SensorModel;
                    RequestPropertyEditor     = PropertyPanelID::TILE_PROPERTY_EDITOR;
                    ImGui::SetWindowFocus( "PROPERTIES" );
                }
            }
            ImGui::TreePop();
        }
        ImGui::PopStyleVar();

        UI::SetCursorPosition( ImGui::GetCursorPos() + ImVec2( 0.0f, 15.0f ) );
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 2 ) );
        if( ImGui::TreeNodeEx( "TILE LAYOUTS", l_Flags ) )
        {
            for( auto &l_TileLayout : SensorModel->mSensorDefinition->mRootLayout.Get<sRelationshipComponent>().mChildren )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
                UI::Text( "{}", ICON_FA_CIRCLE_O );
                UI::SameLine();
                ImGui::PopStyleColor();
                UI::Text( "{}", l_TileLayout.Get<sTag>().mValue );
                UI::SameLine();
                UI::SetCursorPosition( math::vec2( width - 30.0f, UI::GetCurrentCursorPosition().y - 3.0f ) );
                if( EditButton( l_TileLayout, math::vec2{ 20.0, 22.0 } ) )
                {
                    TileInFocus               = l_TileLayout;
                    LayoutEditor.LayoutToEdit = l_TileLayout;
                    LayoutEditor.SensorModel  = SensorModel;
                    RequestPropertyEditor     = PropertyPanelID::TILE_LAYOUT_EDITOR;
                    ImGui::SetWindowFocus( "PROPERTIES" );
                }
            }
            ImGui::TreePop();
        }
        ImGui::PopStyleVar();
    }

} // namespace LTSE::Editor