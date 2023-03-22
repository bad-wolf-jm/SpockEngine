#include "MonoClassHierarchy.h"

#include "UI/CanvasView.h"
#include "UI/TreeNode.h"
#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Core/EntityCollection/Collection.h"
#include "Core/EntityCollection/Components.h"

// #include "OtdrScene/Components.h"

using namespace SE::Core::EntityComponentSystem::Components;

namespace SE::OtdrEditor
{
    void MonoClassHierarchy::DisplayNode( MonoScriptClass &aClass, float aWidth )
    {
        ImGuiTreeNodeFlags lFlags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_FramePadding |
                                    ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
                                    ImGuiTreeNodeFlags_AllowItemOverlap;

        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 3, 3 ) );
        if( aClass.DerivedClasses().size() != 0 )
        {
            ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0.05f, 0.05f, 0.05f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderHovered, ImVec4( 0.025f, 0.025f, 0.025f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderActive, ImVec4( 0.025f, 0.025f, 0.025f, 1.00f ) );

            bool lNodeIsOpen = UI::TreeNodeEx( aClass.FullName().c_str(), lFlags );

            ImGui::PopStyleColor();
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();

            if( lNodeIsOpen )
            {
                for( auto *lDerived : aClass.DerivedClasses() )
                {
                    DisplayNode( *lDerived, 0.0f );
                }
                UI::TreePop();
            }
        }
        else
        {
            ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0.05f, 0.05f, 0.05f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderHovered, ImVec4( 0.025f, 0.025f, 0.025f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_HeaderActive, ImVec4( 0.025f, 0.025f, 0.025f, 1.00f ) );
            ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 0.0f, 2.0f } );

            ImGuiTreeNodeFlags lFlags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_FramePadding |
                                        ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
                                        ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_Leaf;
            bool lNodeIsOpen = UI::TreeNodeEx( aClass.FullName().c_str(), lFlags );

            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();

            UI::TreePop();
        }
        ImGui::PopStyleVar();
    }

    void MonoClassHierarchy::Display( int32_t width, int32_t height )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 5.0f, 0.0f } );
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 0.0f, 0.0f } );

        for( auto &[lName, lClass] : MonoRuntime::GetClasses() )
        {
            if( lClass.ParentClass() == nullptr )
            {
                DisplayNode(lClass, 0.0f);
            }
        }

        ImGui::PopStyleVar();
        ImGui::PopStyleVar();
    }

} // namespace SE::OtdrEditor