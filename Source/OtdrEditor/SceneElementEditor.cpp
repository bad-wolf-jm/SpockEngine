#include "SceneElementEditor.h"

#include "Core/EntityCollection/EditComponent.h"

#include "UI/CanvasView.h"
#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Scene/Components.h"

#include "Scene/Components/VisualHelpers.h"

// using namespace SE::Core::EntityComponentSystem::Components;

namespace SE::OtdrEditor
{

    SceneElementEditor::SceneElementEditor( Ref<VkGraphicContext> aGraphicContext )
        : mGraphicContext{ aGraphicContext } {};

    void SceneElementEditor::Display( int32_t width, int32_t height )
    {
        ImGuiTreeNodeFlags lFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap;

        if( !ElementToEdit ) return;

        char buf[128] = { 0 };
        std::strncpy( buf, ElementToEdit.Get<sTag>().mValue.c_str(),
                      std::min( ElementToEdit.Get<sTag>().mValue.size(), std::size_t( 128 ) ) );
        if( ImGui::InputText( "##TAG_INPUT", buf, ARRAYSIZE( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            ElementToEdit.Get<sTag>().mValue = std::string( buf );
        }
        UI::SameLine();
        if( UI::Button( fmt::format( "{} Add component", ICON_FA_PLUS ).c_str(), math::vec2{ 150.0f, 30.0f } ) )
        {
            ImGui::OpenPopup( "##add_component" );
        }

        if( ImGui::BeginPopup( "##add_component" ) )
        {
            if( ImGui::MenuItem( "HUD Component", NULL, false, !ElementToEdit.Has<sUIComponent>() ) )
            {
                ElementToEdit.Add<sUIComponent>();
            }
            if( ImGui::MenuItem( "Actor Component", NULL, false, !ElementToEdit.Has<sActorComponent>() ) )
            {
                ElementToEdit.Add<sActorComponent>();
            }
            ImGui::EndPopup();
        }

        if( ImGui::CollapsingHeader( "Script", lFlags ) )
        {
            if( ElementToEdit.Has<sActorComponent>() ) EditComponent( ElementToEdit.Get<sActorComponent>() );
        }

        if( ImGui::CollapsingHeader( "HUD", lFlags ) )
        {
            if( ElementToEdit.Has<sUIComponent>() ) EditComponent( ElementToEdit.Get<sUIComponent>() );
        }
    }

} // namespace SE::OtdrEditor