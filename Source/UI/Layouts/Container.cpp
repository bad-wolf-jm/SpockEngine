#include "Container.h"

namespace SE::Core
{
    void UIContainer::PushStyles() {}
    void UIContainer::PopStyles() {}

    ImVec2 UIContainer::RequiredSize()
    {
        float lWidth  = 0.0f;
        float lHeight = 0.0f;

        if( mContent != nullptr ) return mContent->RequiredSize();

        return ImVec2{};
    }

    void UIContainer::SetContent( UIComponent *aChild ) { mContent = aChild; }

    void UIContainer::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        if( !mIsVisible ) return;
        if( mContent == nullptr ) return;

        ImGui::SetCursorPos( aPosition );
        ImGui::PushID( (void *)mContent );
        ImGui::BeginChild( "##ContainerItem", aSize );
        mContent->Update( ImVec2{}, aSize );
        ImGui::EndChild();
        ImGui::PopID();
    }
} // namespace SE::Core