#include "DropdownButton.h"

#include "DotNet/Runtime.h"
namespace SE::Core
{
    UIDropdownButton::UIDropdownButton()
    {
        mImage  = New<UIStackLayout>();
        mText   = New<UILabel>( "" );
        mLayout = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );

        mText->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mLayout->Add( mImage.get(), 40.0f, false, true );
        mLayout->Add( mText.get(), true, true );
        mLayout->SetSimple( true );

        mImage->mIsVisible = false;
    }

    void UIDropdownButton::PushStyles() {}
    void UIDropdownButton::PopStyles() {}

    ImVec2 UIDropdownButton::RequiredSize() { return mLayout->RequiredSize(); }

    void UIDropdownButton::SetContent( UIComponent *aContent ) { mContent = aContent; }
    void UIDropdownButton::SetContentSize( math::vec2 aSize ) { mContentSize = ImVec2{ aSize.x, aSize.y }; }
    void UIDropdownButton::SetText( std::string aText )
    {
        mText->SetText( aText );
        mText->mIsVisible = !( aText.empty() );
    }
    void UIDropdownButton::SetTextColor( math::vec4 aColor ) { mText->SetTextColor( aColor ); }
    void UIDropdownButton::SetImage( UIBaseImage *aValue )
    {
        mImage->Add( aValue, "IMAGE" );
        mImage->mIsVisible = !( aValue == nullptr );
    }

    void UIDropdownButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        if( !mIsVisible ) return;

        ImGuiWindow *window = ImGui::GetCurrentWindow();

        ImVec2 lCurrentScreenPosition = ImGui::GetCursorScreenPos();

        mLayout->Update( aPosition, aSize );

        const ImGuiID id = window->GetID( (void *)this );
        const ImRect  bb( lCurrentScreenPosition, lCurrentScreenPosition + aSize );

        bool hovered, held;
        bool lPressed = ImGui::ButtonBehavior( bb, id, &hovered, &held, ImGuiButtonFlags_MouseButtonLeft );

        if( lPressed ) ImGui::OpenPopup( "##add_component" );

        if( mContent != nullptr )
        {
            ImGui::SetNextWindowPos( ImGui::GetCursorScreenPos() );

            auto lContentSize = ( ( mContentSize.x > 0.0 ) && ( mContentSize.y > 0.0 ) ) ? mContentSize : mContent->RequiredSize();
            ImGui::SetNextWindowSize( lContentSize );
            if( ImGui::BeginPopup( "##add_component" ) )
            {
                mContent->Update( ImVec2{}, lContentSize );
                ImGui::EndPopup();
            }
        }
    }

    void *UIDropdownButton::UIDropdownButton_Create()
    {
        auto lNewImage = new UIDropdownButton();

        return static_cast<void *>( lNewImage );
    }

    void UIDropdownButton::UIDropdownButton_Destroy( void *aInstance ) { delete static_cast<UIDropdownButton *>( aInstance ); }

    void UIDropdownButton::UIDropdownButton_SetContent( void *aInstance, void *aContent )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aInstance );
        auto lContent  = static_cast<UIComponent *>( aContent );

        return lInstance->SetContent( lContent );
    }

    void UIDropdownButton::UIDropdownButton_SetContentSize( void *aInstance, math::vec2 aContentSizse )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aInstance );

        return lInstance->SetContentSize( aContentSizse );
    }

    void UIDropdownButton::UIDropdownButton_SetImage( void *aInstance, void *aImage )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aInstance );
        auto lImage    = static_cast<UIBaseImage *>( aImage );

        lInstance->SetImage( lImage );
    }

    void UIDropdownButton::UIDropdownButton_SetText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UIDropdownButton::UIDropdownButton_SetTextColor( void *aInstance, math::vec4 aColor )
    {
        auto lInstance = static_cast<UIDropdownButton *>( aInstance );

        lInstance->SetTextColor( aColor );
    }
} // namespace SE::Core