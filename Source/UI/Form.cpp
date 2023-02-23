#include "Form.h"

namespace SE::Core
{
    UIForm::UIForm( std::string const &aTitle )
        : mTitle{ aTitle }
    {
    }

    void UIForm::PushStyles() {}
    void UIForm::PopStyles() {}

    void UIForm::SetTitle( std::string const &aTitle ) { mTitle = aTitle; }
    void UIForm::SetContent( UIComponent *aContent ) { mContent = aContent; }

    ImVec2 UIForm::RequiredSize()
    {
        if( mContent != nullptr ) return mContent->RequiredSize();

        return ImVec2{ 100, 100 };
    }

    void UIForm::Update()
    {
        if( !mIsVisible ) return;

        if( ImGui::Begin( mTitle.c_str(), NULL, ImGuiWindowFlags_None ) )
        {
            ImVec2 lContentSize     = ImGui::GetContentRegionAvail();
            ImVec2 lContentPosition = ImGui::GetCursorPos();

            if( mContent != nullptr ) mContent->Update( lContentPosition, lContentSize );
        }
        ImGui::End();
    }

    void UIForm::DrawContent( ImVec2 aPosition, ImVec2 aSize ) {}

} // namespace SE::Core