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

    void UIForm::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetNextWindowPos( aPosition );
        ImGui::SetNextWindowSize( aSize );

        if( ImGui::Begin( mTitle.c_str(), NULL, ImGuiWindowFlags_None ) )
        {
            ImVec2 lContentSize = ImGui::GetContentRegionAvail();

            if( mContent != nullptr ) return mContent->Update( ImVec2{ 0.0f, 0.0f }, lContentSize );
        }
        ImGui::End();
    }

} // namespace SE::Core