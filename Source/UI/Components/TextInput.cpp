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

    void UITextInput::SetHintText( std::string const &aHintText ) { mHintText = aHintText; }
    void UITextInput::SetTextColor( math::vec4 aColor ) { mTextColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }
    void UITextInput::SetBuffersize( uint32_t aSize )
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
        if (mHintText.empty())
            ImGui::InputText( "##input", mBuffer.data(), mBufferSize );
        else
            ImGui::InputTextWithHint("##input", mHintText.c_str(), mBuffer.data(), mBufferSize);
        ImGui::SetCursorPos( aPosition );

        if( lTextColorSet ) ImGui::PopStyleColor();
    }

    void *UITextInput::UITextInput_Create()
    {
        auto lNewTextInput = new UITextInput();

        return static_cast<void *>( lNewTextInput );
    }

    void *UITextInput::UITextInput_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewTextInput = new UITextInput( lString );

        return static_cast<void *>( lNewTextInput );
    }

    void UITextInput::UITextInput_Destroy( void *aInstance ) { delete static_cast<UITextInput *>( aInstance ); }

    void UITextInput::UITextInput_SetHintText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UITextInput *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetHintText( lString );
    }

    void UITextInput::UITextInput_SetTextColor( void *aInstance, math::vec4 *aTextColor )
    {
        auto lInstance = static_cast<UITextInput *>( aInstance );

        lInstance->SetTextColor( *aTextColor );
    }
} // namespace SE::Core