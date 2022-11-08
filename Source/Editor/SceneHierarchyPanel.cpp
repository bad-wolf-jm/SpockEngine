#include "SceneHierarchyPanel.h"

#include "UI/CanvasView.h"
#include "UI/TreeNode.h"
#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Core/EntityRegistry/Components.h"
#include "Core/EntityRegistry/Registry.h"

#include "Scene/Components.h"

using namespace LTSE::Core::EntityComponentSystem::Components;

namespace LTSE::Editor
{

    static bool EditButton( Entity aNode, math::vec2 aSize )
    {
        char lLabel[128];
        sprintf( lLabel, "%s##%d", ICON_FA_PENCIL_SQUARE_O, (uint32_t)aNode );

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0, 0.0, 0.0, 0.0 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0, 1.0, 1.0, 0.02 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0, 1.0, 1.0, 0.02 } );

        bool l_IsVisible;
        bool lDoEdit = UI::Button( lLabel, aSize );

        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        return lDoEdit;
    }

    static bool AddChildButton( Scene::Element aNode, math::vec2 aSize )
    {
        char lLabel[128];
        sprintf( lLabel, "%s##%d", ICON_FA_PLUS, (uint32_t)aNode );

        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0, 0.0, 0.0, 0.0 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0, 1.0, 1.0, 0.02 } );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0, 1.0, 1.0, 0.02 } );

        bool lDoEdit = UI::Button( lLabel, aSize );

        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        return lDoEdit;
    }

    void SceneHierarchyPanel::DisplayNode( Scene::Element aNode, float aWidth )
    {
        ImGuiTreeNodeFlags lFlags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_FramePadding |
                                    ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
                                    ImGuiTreeNodeFlags_AllowItemOverlap;

        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 3 ) );
        if( aNode.Has<sRelationshipComponent>() && ( aNode.Get<sRelationshipComponent>().mChildren.size() != 0 ) &&
            !aNode.Has<LockComponent>() )
        {
            auto        lPos   = UI::GetCurrentCursorPosition();
            std::string lLabel = fmt::format( "{}##node_foo_{}", aNode.Get<sTag>().mValue.c_str(), (uint32_t)aNode );

            ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0.05f, 0.05f, 0.05f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderHovered, ImVec4( 0.025f, 0.025f, 0.025f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderActive, ImVec4( 0.025f, 0.025f, 0.025f, 1.00f ) );

            if( aNode == SelectedElement )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.9f, 0.4f, 0.9f, 1.0f } );
            }
            bool l_NodeIsOpen = UI::TreeNodeEx( lLabel.c_str(), lFlags );
            if( aNode == SelectedElement )
            {
                ImGui::PopStyleColor();
            }
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();

            UI::SameLine();
            UI::SetCursorPosition( math::vec2( aWidth - 45.0f, UI::GetCurrentCursorPosition().y - 0.0f ) );
            if( EditButton( aNode, math::vec2{ 20.0, 22.0 } ) )
            {
                ElementEditor.World         = World;
                ElementEditor.ElementToEdit = aNode;
                RequestEditSceneElement     = true;
                SelectedElement             = aNode;
                ImGui::SetWindowFocus( "PROPERTIES" );
            }
            UI::SameLine();
            if( AddChildButton( aNode, math::vec2{ 20.0, 22.0 } ) ) World->Create( "NEW_ELEMENT", aNode );

            if( l_NodeIsOpen )
            {
                aNode.IfExists<sRelationshipComponent>(
                    [&]( auto &a_Component )
                    {
                        for( auto l_Child : a_Component.mChildren ) DisplayNode( l_Child, aWidth );
                    } );
                UI::TreePop();
            }
        }
        else
        {
            auto lPos = UI::GetCurrentCursorPosition();
            // std::string lLabel = fmt::format( "##leaf_foo_{}", (uint32_t)a_Node );
            std::string lLabel = fmt::format( "{}##node_foo_{}", aNode.Get<sTag>().mValue.c_str(), (uint32_t)aNode );
            ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0.05f, 0.05f, 0.05f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderHovered, ImVec4( 0.025f, 0.025f, 0.025f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderActive, ImVec4( 0.025f, 0.025f, 0.025f, 1.00f ) );
            ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 0.0f, 2.0f } );

            if( aNode == SelectedElement )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.9f, 0.4f, 0.9f, 1.0f } );
            }
            ImGuiTreeNodeFlags lFlags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_FramePadding |
                                        ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
                                        ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_Leaf;
            bool l_NodeIsOpen = UI::TreeNodeEx( lLabel.c_str(), lFlags );
            if( aNode == SelectedElement )
            {
                ImGui::PopStyleColor();
            }

            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();

            UI::SameLine();
            UI::SetCursorPosition( math::vec2( aWidth - 45.0f, UI::GetCurrentCursorPosition().y - 1.0f ) );
            if( EditButton( aNode, math::vec2{ 20.0, 22.0 } ) )
            {
                ElementEditor.World         = World;
                ElementEditor.ElementToEdit = aNode;
                RequestEditSceneElement     = true;
                SelectedElement             = aNode;
            }
            UI::SameLine();
            if( AddChildButton( aNode, math::vec2{ 20.0, 22.0 } ) ) World->Create( "NEW_ELEMENT", aNode );
            UI::TreePop();
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