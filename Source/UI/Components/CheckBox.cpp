#include "CheckBox.h"
#include "DotNet/Runtime.h"
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
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{0.0f, 0.0f});
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
        ImGui::PopStyleVar();
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

    void *UICheckBox::UICheckBox_Create()
    {
        auto lNewLabel = new UICheckBox();

        return static_cast<void *>( lNewLabel );
    }

    void UICheckBox::UICheckBox_Destroy( void *aInstance ) { delete static_cast<UICheckBox *>( aInstance ); }

    void UICheckBox::UICheckBox_OnClick( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UICheckBox *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnClick(
            [lInstance, lDelegate]()
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );
            } );
    }

    bool UICheckBox::UICheckBox_IsChecked( void *aInstance )
    {
        auto lInstance = static_cast<UICheckBox *>( aInstance );

        return lInstance->IsChecked();
    }

    void UICheckBox::UICheckBox_SetIsChecked( void *aInstance, bool aValue )
    {
        auto lInstance = static_cast<UICheckBox *>( aInstance );

        lInstance->SetIsChecked( aValue );
    }

} // namespace SE::Core