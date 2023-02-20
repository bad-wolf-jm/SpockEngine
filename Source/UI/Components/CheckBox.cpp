#include "CheckBox.h"

namespace SE::Core
{
    UICheckBox::UICheckBox( std::function<void()> aOnClick )
        : mOnClick{ aOnClick }
    {
    }

    void UICheckBox::PushStyles() {}
    void UICheckBox::PopStyles() {}

    void UICheckBox::OnClick( std::function<void()> aOnClick ) { mOnClick = aOnClick; }

    void UICheckBox::PushStyles( bool aEnabled )
    {
        if( !aEnabled )
        {
            ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.3f, 0.3f, 0.3f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
        }
    }

    void UICheckBox::PopStyles( bool aEnabled )
    {
        if( !aEnabled ) ImGui::PopStyleColor( 4 );
    }

    ImVec2 UICheckBox::RequiredSize()
    {
        const float lSize = ImGui::GetFrameHeight();

        return ImVec2{ lSize, lSize };
    }

    void UICheckBox::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        PushStyles( lEnabled );

        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, RequiredSize(), aSize ) );

        if( ImGui::Checkbox( "", &mIsChecked ) && mOnClick && lEnabled ) mOnClick();

        PopStyles( lEnabled );
    }

} // namespace SE::Core