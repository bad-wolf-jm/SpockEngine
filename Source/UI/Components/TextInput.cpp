#include "TextInput.h"

namespace SE::Core
{
    UITextInput::UITextInput( string_t const &aHintText )
        : mHintText{ aHintText }
    {
    }

    void UITextInput::PushStyles() {}
    void UITextInput::PopStyles() {}

    void UITextInput::OnTextChanged( std::function<void( string_t )> aOnTextChanged ) { mOnTextChanged = aOnTextChanged; }

    string_t &UITextInput::GetText() { return mBuffer; }
    void      UITextInput::SetHintText( string_t const &aHintText ) { mHintText = aHintText; }
    void      UITextInput::SetTextColor( math::vec4 aColor ) { mTextColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }
    void      UITextInput::SetBuffersize( uint32_t aSize )
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
} // namespace SE::Core