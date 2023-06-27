#include "ColorButton.h"
#include "Engine/Engine.h"

namespace SE::Core
{
    void UIColorButton::PushStyles() {}
    void UIColorButton::PopStyles() {}

    ImVec2 UIColorButton::RequiredSize() { return ImVec2{ 30, 30 }; }

    void UIColorButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::ColorEdit4( "##", (float *)&mColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel );
    }
} // namespace SE::Core