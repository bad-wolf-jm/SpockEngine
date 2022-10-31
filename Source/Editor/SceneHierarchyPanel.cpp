#include "SceneHierarchyPanel.h"

#include "UI/CanvasView.h"
#include "UI/UI.h"
#include "UI/Widgets.h"
#include "UI/TreeNode.h"

#include "Core/EntityRegistry/Components.h"
#include "Core/EntityRegistry/Registry.h"

#include "Scene/Components.h"

using namespace LTSE::Core::EntityComponentSystem::Components;

namespace LTSE::Editor
{

    static bool EditButton( Entity a_Node, math::vec2 a_Size )
    {
        char l_OnLabel[128];
        sprintf( l_OnLabel, "%s##%d", ICON_FA_PENCIL_SQUARE_O, (uint32_t)a_Node );

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0, 0.0, 0.0, 0.0 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0, 1.0, 1.0, 0.02 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0, 1.0, 1.0, 0.02 } );

        bool l_IsVisible;
        bool l_DoEdit = UI::Button( l_OnLabel, a_Size );
        // UI::Button( l_OnLabel, a_Size, [&]() { l_DoEdit = true; } );

        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        return l_DoEdit;
    }

    static bool AddChildButton( Scene::Element a_Node, math::vec2 a_Size )
    {
        char l_OnLabel[128];
        sprintf( l_OnLabel, "%s##%d", ICON_FA_PLUS, (uint32_t)a_Node );

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0, 0.0, 0.0, 0.0 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0, 1.0, 1.0, 0.02 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0, 1.0, 1.0, 0.02 } );

        bool l_DoEdit = UI::Button( l_OnLabel, a_Size );

        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        return l_DoEdit;
    }

    void SceneHierarchyPanel::DisplayNode( Scene::Element a_Node, float a_Width )
    {
        ImGuiTreeNodeFlags l_Flags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
                                     ImGuiTreeNodeFlags_AllowItemOverlap;
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 3 ) );
        if( a_Node.Has<sRelationshipComponent>() && ( a_Node.Get<sRelationshipComponent>().mChildren.size() != 0 ) && !a_Node.Has<LockComponent>() )
        {
            auto l_Pos          = UI::GetCurrentCursorPosition();
            std::string l_Label = fmt::format( "##node_foo_{}", (uint32_t)a_Node );

            ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0.05f, 0.05f, 0.05f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderHovered, ImVec4( 0.07f, 0.07f, 0.07f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderActive, ImVec4( 0.07f, 0.07f, 0.07f, 1.0f ) );
            bool l_NodeIsOpen = UI::TreeNodeEx( l_Label.c_str(), l_Flags );
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();

            UI::SetCursorPosition( l_Pos + math::vec2( 16.0f, 3.0f ) );

            ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
            UI::Text( "{}", ICON_FA_CIRCLE );
            UI::SameLine();
            ImGui::PopStyleColor();

            if( a_Node == SelectedElement )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.9f, 0.4f, 0.9f, 1.0f } );
            }
            UI::Text( a_Node.Get<sTag>().mValue );
            if( a_Node == SelectedElement )
            {
                ImGui::PopStyleColor();
            }

            UI::SameLine();
            UI::SetCursorPosition( math::vec2( a_Width - 45.0f, UI::GetCurrentCursorPosition().y - 4.0f ) );
            if( EditButton( a_Node, math::vec2{ 20.0, 22.0 } ) )
            {
                ElementEditor.World         = World;
                ElementEditor.ElementToEdit = a_Node;
                RequestEditSceneElement     = true;
                SelectedElement             = a_Node;
                ImGui::SetWindowFocus( "PROPERTIES" );
            }
            UI::SameLine();
            if( AddChildButton( a_Node, math::vec2{ 20.0, 22.0 } ) )
                World->Create( "NEW_ELEMENT", a_Node );

            if( l_NodeIsOpen )
            {
                a_Node.IfExists<sRelationshipComponent>(
                    [&]( auto &a_Component )
                    {
                        for( auto l_Child : a_Component.mChildren )
                            DisplayNode( l_Child, a_Width );
                    } );
                ImGui::TreePop();
            }
        }
        else
        {
            auto l_Pos          = UI::GetCurrentCursorPosition();
            std::string l_Label = fmt::format( "##leaf_foo_{}", (uint32_t)a_Node );
            ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0.05f, 0.05f, 0.05f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderHovered, ImVec4( 0.07f, 0.07f, 0.07f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderActive, ImVec4( 0.07f, 0.07f, 0.07f, 1.0f ) );
            ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 0.0f, 2.0f } );
            if( ImGui::Selectable( l_Label.c_str(), false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap, ImVec2{ 0.0f, 17.0f } ) )
            {
                ElementEditor.World         = World;
                ElementEditor.ElementToEdit = a_Node;
                RequestEditSceneElement     = true;
                SelectedElement             = a_Node;
            }
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();
            UI::SetCursorPosition( l_Pos + math::vec2{ 12.0f, 0.0f } );

            ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.5f, 0.2f, 0.7f, 1.0f } );
            ImGui::AlignTextToFramePadding();
            UI::Text( "{}", ICON_FA_CIRCLE_O );
            UI::SameLine();
            ImGui::PopStyleColor();

            if( a_Node == SelectedElement )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.9f, 0.4f, 0.9f, 1.0f } );
            }

            UI::Text( a_Node.Get<sTag>().mValue );
            if( a_Node == SelectedElement )
            {
                ImGui::PopStyleColor();
            }

            UI::SameLine();
            UI::SetCursorPosition( math::vec2( a_Width - 45.0f, UI::GetCurrentCursorPosition().y - 3.0f ) );
            if( EditButton( a_Node, math::vec2{ 20.0, 22.0 } ) )
            {
                ElementEditor.World         = World;
                ElementEditor.ElementToEdit = a_Node;
                RequestEditSceneElement     = true;
                SelectedElement             = a_Node;
            }
            UI::SameLine();
            if( AddChildButton( a_Node, math::vec2{ 20.0, 22.0 } ) )
                World->Create( "NEW_ELEMENT", a_Node );
        }
        ImGui::PopStyleVar();
    }

    void SceneHierarchyPanel::Display( int32_t width, int32_t height )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 5.0f, 0.0f } );
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 0.0f, 0.0f } );
        DisplayNode( World->Root, UI::GetAvailableContentSpace().x );
        ImGui::PopStyleVar();
        ImGui::PopStyleVar();
    }

} // namespace LTSE::Editor