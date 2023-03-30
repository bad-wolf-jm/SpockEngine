#include "Dialog.h"

namespace SE::Core
{
    UIDialog::UIDialog( std::string aTitle, math::vec2 aSize )
        : mTitle{ aTitle }
        , mSize{ aSize }
    {
    }

    void UIDialog::PushStyles() {}
    void UIDialog::PopStyles() {}

    void UIDialog::SetTitle( std::string const &aTitle ) { mTitle = aTitle; }
    void UIDialog::SetSize( math::vec2 aSize ) { mSize = aSize; }
    void UIDialog::SetContent( UIComponent *aContent ) { mContent = aContent; }

    void UIDialog::Open() { ImGui::OpenPopup( mTitle.c_str() ); }

    void UIDialog::Update()
    {
        ImGui::SetNextWindowSize( ImVec2{ mSize.x, mSize.y } );
        ImGuiWindowFlags lFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoResize |
                                  ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2{} );
        if( ImGui::BeginPopupModal( mTitle.c_str(), nullptr, lFlags ) )
        {
            ImVec2 lContentSize     = ImGui::GetContentRegionAvail();
            ImVec2 lContentPosition = ImGui::GetCursorPos();

            if( mContent != nullptr ) mContent->Update( lContentPosition, lContentSize );

            ImGui::EndPopup();
        }
        ImGui::PopStyleVar();
    }

    void UIDialog::DrawContent( ImVec2 aPosition, ImVec2 aSize ) {}

} // namespace SE::Core