#include "TileFlashEditor.h"

#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

#include "LidarSensorModel/Components.h"

using namespace LTSE::Core;
using namespace LTSE::SensorModel;

namespace LTSE::Editor
{

    FlashAttenuationBindingPopup::FlashAttenuationBindingPopup( std::string a_Title, math::vec2 a_Size )
        : PopupWindow( a_Title, a_Size )
    {
    }

    void FlashAttenuationBindingPopup::WindowContent()
    {
        if( !Visible )
            return;
        ImVec2 l_PopupSize = ImGui::GetWindowSize();

        ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
                                ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;
        ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, ImVec2( 5, 5 ) );

        std::array<std::string, 9> lColumns = { "ID", "OffsetX", "OffsetY", "SizeX", "SizeY", "Photodetector", "Laser", "Diffusion", "Reduction" };

        if( ImGui::BeginTable( "##TABLE", lColumns.size(), flags, ImVec2{ l_PopupSize.x - 20.0f, l_PopupSize.y - 150.0f } ) )
        {
            ImGui::TableSetupScrollFreeze( 1, 1 );
            for( auto &a_ColumnName : lColumns )
                ImGui::TableSetupColumn( a_ColumnName.c_str() );
            ImGui::TableHeadersRow();

            for( auto &lFlash : TileToEdit.Get<sRelationshipComponent>().mChildren )
            {
                ImGui::TableNextRow();

                auto &l_FlashSpec     = lFlash.Get<sLaserFlashSpecificationComponent>();
                auto &l_FlashPosition = l_FlashSpec.mPosition;

                ImGui::TableSetColumnIndex( 0 );
                UI::Text( "{}", l_FlashSpec.mFlashID );

                ImGui::TableSetColumnIndex( 1 );
                UI::Text( "{:.4f}", l_FlashSpec.mPosition.x );

                ImGui::TableSetColumnIndex( 2 );
                UI::Text( "{:.4f}", l_FlashSpec.mPosition.y );

                ImGui::TableSetColumnIndex( 3 );
                UI::Text( "{:.4f}", l_FlashSpec.mExtent.x * 2.0f );

                ImGui::TableSetColumnIndex( 4 );
                UI::Text( "{:.4f}", l_FlashSpec.mExtent.x * 2.0f );

                ImGui::TableSetColumnIndex( 5 );
                if( lFlash.Has<sJoinComponent<sPhotoDetector>>() && lFlash.Get<sJoinComponent<sPhotoDetector>>().mJoinEntity )
                {
                    UI::Text( "{}", lFlash.Get<sJoinComponent<sPhotoDetector>>().mJoinEntity.Get<sSensorComponent>().mID );
                }
                else
                {
                    UI::Text( "N/A" );
                }

                ImGui::TableSetColumnIndex( 6 );
                if( lFlash.Has<sJoinComponent<sLaserAssembly>>() && lFlash.Get<sJoinComponent<sLaserAssembly>>().mJoinEntity )
                {
                    UI::Text( "{}", lFlash.Get<sJoinComponent<sLaserAssembly>>().mJoinEntity.Get<sSensorComponent>().mID );
                }
                else
                {
                    UI::Text( "N/A" );
                }

                ImGui::TableSetColumnIndex( 7 );
                if( lFlash.Has<sJoinComponent<sDiffusionPattern>>() && lFlash.Get<sJoinComponent<sDiffusionPattern>>().mJoinEntity )
                {
                    auto lAssetData = lFlash.Get<sJoinComponent<sDiffusionPattern>>().mJoinEntity;
                    if( lAssetData.Has<sInternalAssetReference>() )
                    {
                        auto &lAssetReference = lAssetData.Get<sInternalAssetReference>();
                        UI::Text( "{}[{}]", lAssetReference.mParentID, lAssetReference.mID );
                    }
                    else
                    {
                        UI::Text( "Builtin" );
                    }
                }
                else
                {
                    UI::Text( "N/A" );
                }

                ImGui::TableSetColumnIndex( 8 );
                UI::Text( "N/A" );
            }

            ImGui::EndTable();
        }
        ImGui::PopStyleVar();

        UI::SetCursorPosition( l_PopupSize - ImVec2{ 150, 40 } - ImVec2{ 15, 15 } );
        if( ImGui::Button( "Cancel##FlashAttenuationBindingPopup", ImVec2{ 150, 40 } ) )
        {
            ImGui::CloseCurrentPopup();
            Visible               = false;
            TileToEdit            = Entity{};
            DiffusionPatternAsset = Entity{};
        }
    }

} // namespace LTSE::Editor