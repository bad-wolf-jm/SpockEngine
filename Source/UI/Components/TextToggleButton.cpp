#include "TextToggleButton.h"
#include "DotNet/Runtime.h"

namespace SE::Core
{
    UITextToggleButton::UITextToggleButton( std::string const &aText )
        : UILabel{ aText }
    {
    }

    UITextToggleButton::UITextToggleButton( std::string const &aText, std::function<bool( bool )> aOnChange )
        : UILabel{ aText }
        , mOnChange{ aOnChange }
    {
    }

    void UITextToggleButton::PushStyles() {}
    void UITextToggleButton::PopStyles() {}

    void UITextToggleButton::OnChange( std::function<bool( bool )> aOnChange ) { mOnChange = aOnChange; }

    bool UITextToggleButton::IsActive() { return mActivated; }
    void UITextToggleButton::SetActive( bool aValue ) { mActivated = aValue; }

    // void UITextToggleButton::SetText( std::string const &aText ) { UILabel::SetText( aText ); }

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
        if( ImGui::Button( mText.c_str(), aSize ) && mOnChange && lEnabled ) mActivated = mOnChange( mActivated );
        ImGui::PopStyleColor();

        PopStyles( lEnabled );
    }

    void *UITextToggleButton::UITextToggleButton_Create()
    {
        auto lNewButton = new UITextToggleButton();

        return static_cast<void *>( lNewButton );
    }

    void *UITextToggleButton::UITextToggleButton_CreateWithText( void *aText )
    {
        auto lString    = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewButton = new UITextToggleButton( lString );

        return static_cast<void *>( lNewButton );
    }

    void UITextToggleButton::UITextToggleButton_Destroy( void *aInstance ) { delete static_cast<UITextToggleButton *>( aInstance ); }

    bool UITextToggleButton::UITextToggleButton_IsActive( void *aInstance )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aInstance );

        return lInstance->IsActive();
    }

    void UITextToggleButton::UITextToggleButton_SetActive( void *aInstance, bool aValue )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aInstance );

        lInstance->SetActive( aValue );
    }

    void UITextToggleButton::UITextToggleButton_SetActiveColor( void *aInstance, math::vec4 *aColor )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aInstance );

        lInstance->SetActiveColor( *aColor );
    }

    void UITextToggleButton::UITextToggleButton_SetInactiveColor( void *aInstance, math::vec4 *aColor )
    {
        auto lInstance = static_cast<UITextToggleButton *>( aInstance );

        lInstance->SetInactiveColor( *aColor );
    }

} // namespace SE::Core