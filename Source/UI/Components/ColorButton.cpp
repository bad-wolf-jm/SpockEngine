#include "ColorButton.h"
#include "Engine/Engine.h"

#include "DotNet/Runtime.h"

namespace SE::Core
{
    void UIColorButton::PushStyles() {}
    void UIColorButton::PopStyles() {}

    ImVec2 UIColorButton::RequiredSize()
    {
        return ImVec2{30, 30};
    }

    void UIColorButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::ColorEdit4("##", (float*)&mColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
    }

    void *UIColorButton::UIColorButton_Create()
    {
        auto lNewLabel = new UIColorButton();

        return static_cast<void *>( lNewLabel );
    }

    void UIColorButton::UIColorButton_Destroy( void *aInstance ) { delete static_cast<UIColorButton *>( aInstance ); }
} // namespace SE::Core