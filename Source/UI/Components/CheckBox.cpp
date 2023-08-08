#include "CheckBox.h"

namespace SE::Core
{
    UICheckBox::UICheckBox( std::function<void()> aOnClick )
        : mOnClick{ aOnClick }
    {
    }

    void UICheckBox::PushStyles()
    {
    }
    void UICheckBox::PopStyles()
    {
    }

    void UICheckBox::OnClick( std::function<void()> aOnClick )
    {
        mOnClick = aOnClick;
    }

    void UICheckBox::PushStyles( bool aEnabled )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2{ 0.0f, 0.0f } );
        ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, 4.0f );

        if( !aEnabled )
        {
            ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 1.f, 1.f, 1.f, .5f } );
            ImGui::PushStyleColor( ImGuiCol_FrameBgActive, ImVec4{ 0.1f, 0.1f, 0.1f, .1f } );
            ImGui::PushStyleColor( ImGuiCol_FrameBgHovered, ImVec4{ 0.1f, 0.1f, 0.1f, .1f } );
            ImGui::PushStyleColor( ImGuiCol_FrameBg, ImVec4{ 0.1f, 0.1f, 0.1f, .1f } );
        }
        else
        {
            ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 1.f, 1.f, 1.f, 1.f } );
            ImGui::PushStyleColor( ImGuiCol_FrameBgActive, ImVec4{ 0.1f, 0.1f, 0.1f, .25f } );
            ImGui::PushStyleColor( ImGuiCol_FrameBgHovered, ImVec4{ 0.1f, 0.1f, 0.1f, .25f } );
            ImGui::PushStyleColor( ImGuiCol_FrameBg, ImVec4{ 0.1f, 0.1f, 0.1f, .25f } );
        }
    }

    void UICheckBox::PopStyles( bool aEnabled )
    {
        ImGui::PopStyleColor( 4 );
        ImGui::PopStyleVar(2);
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
        if( ImGui::Checkbox( "", &mIsChecked ) && mOnClick && lEnabled )
        {
            mOnClick();
        }

        PopStyles( lEnabled );
    }
} // namespace SE::Core