#include "Form.h"
#include "DotNet/Runtime.h"

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

    void *UIForm::UIForm_Create()
    {
        auto lNewForm = new UIForm();

        return static_cast<void *>( lNewForm );
    }

    void UIForm::UIForm_Destroy( void *aInstance ) { delete static_cast<UIForm *>( aInstance ); }

    void UIForm::UIForm_SetTitle( void *aInstance, void *aTitle )
    {
        auto lInstance = static_cast<UIForm *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );

        lInstance->SetTitle( lString );
    }

    void UIForm::UIForm_SetContent( void *aInstance, void *aContent )
    {
        auto lInstance = static_cast<UIForm *>( aInstance );
        auto lContent  = static_cast<UIComponent *>( aContent );

        lInstance->SetContent( lContent );
    }

    void UIForm::UIForm_Update( void *aInstance )
    {
        auto lInstance = static_cast<UIForm *>( aInstance );

        lInstance->Update();
    }

} // namespace SE::Core