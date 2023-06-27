#include "Slider.h"
#include "Engine/Engine.h"

namespace SE::Core
{
    void UISlider::PushStyles()
    {
        ImGui::PushStyleColor( ImGuiCol_FrameBg, ImVec4{ .03f, 0.03f, 0.03f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_FrameBgActive, ImVec4{ .04f, 0.04f, 0.04f, 1.0f } );
        ImGui::PushStyleColor( ImGuiCol_FrameBgHovered, ImVec4{ .04f, 0.04f, 0.04f, 1.0f } );
    }

    void UISlider::PopStyles() { ImGui::PopStyleColor( 3 ); }

    ImVec2 UISlider::RequiredSize() { return ImVec2{ 30, 30 }; }

    void UISlider::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetNextItemWidth( aSize.x );
        ImGui::SliderFloat( "", &mValue, 0.0f, 1.0f, "" );
    }
} // namespace SE::Core