#include "Button.h"

namespace SE::Core
{
    void UIButton::PushStyles() {}
    void UIButton::PopStyles() {}

    void UIButton::PushStyles( bool aEnabled )
    {
        if( !aEnabled )
        {
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } );
        }
    }

    void UIButton::PopStyles( bool aEnabled )
    {
        if( !aEnabled ) ImGui::PopStyleColor( 3 );
    }

    ImVec2 UIButton::RequiredSize()
    {
        PushStyles( mIsEnabled );

        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        PopStyles( mIsEnabled );

        return lTextSize + ImGui::GetStyle().FramePadding * 2.0f;
    }

    void UIButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        PushStyles( lEnabled );

        ImGui::SetCursorPos( aPosition );

        if( ImGui::Button( mText.c_str(), aSize ) && mOnClick && lEnabled ) mOnClick();

        PopStyles( lEnabled );
    }

} // namespace SE::Core