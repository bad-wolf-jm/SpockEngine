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

    // void *UIContainer::UIContainer_Create()
    // {
    //     auto lNewLayout = new UIContainer();

    //     return static_cast<void *>( lNewLayout );
    // }

    // void UIContainer::UIContainer_Destroy( void *aInstance ) { delete static_cast<UIContainer *>( aInstance ); }

    // void UIContainer::UIContainer_SetContent( void *aInstance, void *aChild )
    // {
    //     auto lInstance = static_cast<UIContainer *>( aInstance );
    //     auto lChild    = static_cast<UIComponent *>( aChild );

    //     lInstance->SetContent( lChild );
    // }
} // namespace SE::Core