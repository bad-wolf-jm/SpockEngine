#include "UIComponent.h"

namespace SE::Core
{
    void UIComponent::Update( ImVec2 aPosition, ImVec2 aSize )
    {
        if( !mIsVisible ) return;

        ImGui::PushID( (void *)this );

        PushStyles();
        DrawContent( aPosition, aSize );
        PopStyles();

        ImGui::PopID();

        if( !mAllowDragDrop || !mIsEnabled ) return;
    }
} // namespace SE::Core
