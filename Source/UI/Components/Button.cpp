#include "Button.h"
#include "Mono/MonoRuntime.h"
namespace SE::Core
{
    UIButton::UIButton( std::string const &aText )
        : UILabel{ aText }
    {
    }

    UIButton::UIButton( std::string const &aText, std::function<void()> aOnClick )
        : UILabel{ aText }
        , mOnClick{ aOnClick }
    {
    }

    void UIButton::PushStyles() {}
    void UIButton::PopStyles() {}

    void UIButton::OnClick( std::function<void()> aOnClick ) { mOnClick = aOnClick; }

    void UIButton::SetText( std::string const &aText ) { UILabel::SetText( aText ); }

    void UIButton::PushStyles( bool aEnabled )
    {
        if( !aEnabled )
        {
            ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.3f, 0.3f, 0.3f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
        }
    }

    void UIButton::PopStyles( bool aEnabled )
    {
        if( !aEnabled ) ImGui::PopStyleColor( 4 );
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


    void *UIButton::UIButton_Create()
    {
        auto lNewButton = new UIButton();

        return static_cast<void *>( lNewButton );
    }

    void *UIButton::UIButton_CreateWithText( void *aText )
    {
        auto lString   = MonoRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewButton = new UIButton( lString );

        return static_cast<void *>( lNewButton );
    }

    void UIButton::UIButton_Destroy( void *aInstance ) { delete static_cast<UILabel *>( aInstance ); }

    void UIButton::UIButton_SetText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UILabel *>( aInstance );
        auto lString   = MonoRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UIButton::UIButton_OnClick( void *aInstance, math::vec4 *aTextColor )
    {
        auto lInstance = static_cast<UILabel *>( aInstance );

        lInstance->SetTextColor( *aTextColor );
    }
} // namespace SE::Core