#include "TextInput.h"

#include "DotNet/Runtime.h"

namespace SE::Core
{
    UITextInput::UITextInput( std::string const &aHintText )
        : mHintText{ aHintText }
    {
    }

    void UITextInput::PushStyles() {}
    void UITextInput::PopStyles() {}

    void UITextInput::OnTextChanged( std::function<void( std::string )> aOnTextChanged ) { mOnTextChanged = aOnTextChanged; }

    std::string &UITextInput::GetText() { return mBuffer; }
    void         UITextInput::SetHintText( std::string const &aHintText ) { mHintText = aHintText; }
    void         UITextInput::SetTextColor( math::vec4 aColor ) { mTextColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }
    void         UITextInput::SetBuffersize( uint32_t aSize )
    {
        mBufferSize = aSize;
        mBuffer.reserve( mBufferSize );
    }

    ImVec2 UITextInput::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mHintText.c_str() );

        return lTextSize;
    }

    void UITextInput::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet ) ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        auto lTextSize     = ImGui::CalcTextSize( mHintText.c_str() );
        auto lTextPosition = GetContentAlignedposition( mHAlign, mVAlign, aPosition, aSize, aSize );

        ImGui::SetCursorPos( lTextPosition );
        bool lTextChanged = false;
        if( mHintText.empty() )
            lTextChanged = ImGui::InputText( "##input", mBuffer.data(), mBufferSize, ImGuiInputTextFlags_EnterReturnsTrue );
        else
            lTextChanged = ImGui::InputTextWithHint( "##input", mHintText.c_str(), mBuffer.data(), mBufferSize,
                                                     ImGuiInputTextFlags_EnterReturnsTrue );
        ImGui::SetCursorPos( aPosition );

        if( lTextChanged && mOnTextChanged && mIsEnabled ) mOnTextChanged( mBuffer );

        if( lTextColorSet ) ImGui::PopStyleColor();
    }

    // void *UITextInput::UITextInput_Create()
    // {
    //     auto lNewTextInput = new UITextInput();

    //     return static_cast<void *>( lNewTextInput );
    // }

    // void *UITextInput::UITextInput_CreateWithText( void *aText )
    // {
    //     auto lString       = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
    //     auto lNewTextInput = new UITextInput( lString );

    //     return static_cast<void *>( lNewTextInput );
    // }

    // void UITextInput::UITextInput_Destroy( void *aInstance ) { delete static_cast<UITextInput *>( aInstance ); }

    // void UITextInput::UITextInput_SetHintText( void *aInstance, void *aText )
    // {
    //     auto lInstance = static_cast<UITextInput *>( aInstance );
    //     auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

    //     lInstance->SetHintText( lString );
    // }

    // void *UITextInput::UITextInput_GetText( void *aInstance )
    // {
    //     auto lInstance = static_cast<UITextInput *>( aInstance );

    //     return DotNetRuntime::NewString( lInstance->GetText() );
    // }

    // void UITextInput::UITextInput_SetTextColor( void *aInstance, math::vec4 *aTextColor )
    // {
    //     auto lInstance = static_cast<UITextInput *>( aInstance );

    //     lInstance->SetTextColor( *aTextColor );
    // }

    // void UITextInput::UITextInput_SetBufferSize( void *aInstance, uint32_t aBufferSize )
    // {
    //     auto lInstance = static_cast<UITextInput *>( aInstance );

    //     lInstance->SetBuffersize( aBufferSize );
    // }

    // void UITextInput::UITextInput_OnTextChanged( void *aInstance, void *aDelegate )
    // {
    //     auto lInstance = static_cast<UITextInput *>( aInstance );
    //     auto lDelegate = static_cast<MonoObject *>( aDelegate );

    //     if( lInstance->mOnTextChangedDelegate != nullptr ) mono_gchandle_free( lInstance->mOnTextChangedDelegateHandle );

    //     lInstance->mOnTextChangedDelegate       = aDelegate;
    //     lInstance->mOnTextChangedDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

    //     lInstance->OnTextChanged(
    //         [lInstance, lDelegate]( std::string aText )
    //         {
    //             auto lDelegateClass = mono_object_get_class( lDelegate );
    //             auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

    //             auto  lString   = DotNetRuntime::NewString( aText );
    //             void *lParams[] = { (void *)lString };
    //             auto  lValue    = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
    //             mono_free( lString );
    //         } );
    // }

} // namespace SE::Core