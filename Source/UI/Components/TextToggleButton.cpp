#include "TextToggleButton.h"

namespace SE::Core
{
    UITextToggleButton::UITextToggleButton( string_t const &aText )
        : UILabel{ aText }
    {
    }

    UITextToggleButton::UITextToggleButton( string_t const &aText, std::function<bool( bool )> aOnChange )
        : UILabel{ aText }
        , mOnClicked{ aOnChange }
    {
    }

    void UITextToggleButton::PushStyles()
    {
    }
    void UITextToggleButton::PopStyles()
    {
    }

    void UITextToggleButton::OnClick( std::function<bool( bool )> aOnChange )
    {
        mOnClicked = aOnChange;
    }
    void UITextToggleButton::OnChanged( std::function<void()> aOnChanged )
    {
        mOnChanged = aOnChanged;
    }

    bool UITextToggleButton::IsActive()
    {
        return mActivated;
    }
    void UITextToggleButton::SetActive( bool aValue )
    {
        mActivated = aValue;
    }

    void UITextToggleButton::SetActiveColor( math::vec4 const &aColor )
    {
        mActiveColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w };
    }

    void UITextToggleButton::SetInactiveColor( math::vec4 const &aColor )
    {
        mInactiveColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w };
    }

    void UITextToggleButton::PushStyles( bool aEnabled )
    {
        if( !aEnabled )
        {
            ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.3f, 0.3f, 0.3f, .2f } );

            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
        }
        else
        {
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0f, 1.0f, 1.0f, 0.01f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0f, 1.0f, 1.0f, 0.02f } );
        }
    }

    void UITextToggleButton::PopStyles( bool aEnabled )
    {
        if( !aEnabled )
            ImGui::PopStyleColor( 4 );
        else
            ImGui::PopStyleColor( 3 );
    }

    ImVec2 UITextToggleButton::RequiredSize()
    {
        PushStyles( mIsEnabled );

        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        PopStyles( mIsEnabled );

        return lTextSize + ImGui::GetStyle().FramePadding * 2.0f;
    }

    void UITextToggleButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        PushStyles( lEnabled );

        ImGui::SetCursorPos( aPosition );

        ImGui::PushStyleColor( ImGuiCol_Text, mActivated ? mActiveColor : mInactiveColor );
        if( ImGui::Button( mText.c_str(), aSize ) && mOnClicked && lEnabled )
            mActivated = mOnClicked( mActivated );
        ImGui::PopStyleColor();

        PopStyles( lEnabled );
    }
} // namespace SE::Core