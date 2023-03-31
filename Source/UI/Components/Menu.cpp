#include "Menu.h"

#include "DotNet/Runtime.h"
namespace SE::Core
{
    UIMenuItem::UIMenuItem( std::string const &aText )
        : mText{ aText }
    {
    }

    UIMenuItem::UIMenuItem( std::string const &aText, std::string const &aShortcut )
        : mText{ aText }
        , mShortcut{ aShortcut }
    {
    }

    void UIMenuItem::PushStyles() {}
    void UIMenuItem::PopStyles() {}

    void UIMenuItem::SetText( std::string const &aText ) { mText = aText; }
    void UIMenuItem::SetShortcut( std::string const &aShortcut ) { mShortcut = aShortcut; }
    void UIMenuItem::SetTextColor( math::vec4 aColor ) { mTextColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }

    ImVec2 UIMenuItem::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UIMenuItem::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet ) ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        const char *lShortcut = mShortcut.empty() ? nullptr : mShortcut.c_str();

        bool lEnabled = mIsEnabled;

        bool lSelected = false;
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 30.0f, 10.0f ) );
        if( ImGui::MenuItem( mText.c_str(), lShortcut, &lSelected ) && mOnTrigger && lEnabled ) mOnTrigger();
        ImGui::PopStyleVar();

        if( lTextColorSet ) ImGui::PopStyleColor();
    }

    void *UIMenuItem::UIMenuItem_Create()
    {
        auto lNewLabel = new UIMenuItem();

        return static_cast<void *>( lNewLabel );
    }

    void *UIMenuItem::UIMenuItem_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UIMenuItem( lString );

        return static_cast<void *>( lNewLabel );
    }

    void *UIMenuItem::UIMenuItem_CreateWithTextAndShortcut( void *aText, void *aShortcut )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lShortcut = DotNetRuntime::NewString( static_cast<MonoString *>( aShortcut ) );
        auto lNewLabel = new UIMenuItem( lString, lShortcut );

        return static_cast<void *>( lNewLabel );
    }

    void UIMenuItem::UIMenuItem_Destroy( void *aInstance ) { delete static_cast<UIMenuItem *>( aInstance ); }

    void UIMenuItem::UIMenuItem_SetText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UIMenuItem *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UIMenuItem::UIMenuItem_SetShortcut( void *aInstance, void *aShortcut )
    {
        auto lInstance = static_cast<UIMenuItem *>( aInstance );
        auto lShortcut = DotNetRuntime::NewString( static_cast<MonoString *>( aShortcut ) );

        lInstance->SetShortcut( lShortcut );
    }

    void UIMenuItem::UIMenuItem_SetTextColor( void *aInstance, math::vec4 *aTextColor )
    {
        auto lInstance = static_cast<UIMenuItem *>( aInstance );

        lInstance->SetTextColor( *aTextColor );
    }

    void UIMenuItem::UIMenuItem_OnTrigger( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIMenuItem *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnTriggerDelegate != nullptr ) mono_gchandle_free( lInstance->mOnTriggerDelegateHandle );

        lInstance->mOnTriggerDelegate       = aDelegate;
        lInstance->mOnTriggerDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnTrigger(
            [lInstance, lDelegate]()
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );
            } );
    }

    UIMenu::UIMenu( std::string const &aText )
        : UIMenuItem( aText )
    {
    }

    void UIMenu::PushStyles() {}
    void UIMenu::PopStyles() {}

    ImVec2 UIMenu::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UIMenu::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet ) ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        if( ImGui::BeginMenu( mText.c_str(), &mIsEnabled ) )
        {
            for( auto &lItem : mActions ) lItem->Update( aPosition, aSize );

            ImGui::EndMenu();
        }

        if( lTextColorSet ) ImGui::PopStyleColor();
    }

    void *UIMenu::UIMenu_Create()
    {
        auto lNewLabel = new UIMenu();

        return static_cast<void *>( lNewLabel );
    }

    void *UIMenu::UIMenu_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UIMenu( lString );

        return static_cast<void *>( lNewLabel );
    }

    void UIMenu::UIMenu_Destroy( void *aInstance ) { delete static_cast<UIMenu *>( aInstance ); }
} // namespace SE::Core