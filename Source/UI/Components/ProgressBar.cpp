#include "ProgressBar.h"
#include "Engine/Engine.h"



namespace SE::Core
{
    void UIProgressBar::PushStyles() {}
    void UIProgressBar::PopStyles() {}

    void UIProgressBar::SetText( string_t const &aText ) { mText = aText; }
    void UIProgressBar::SetTextColor( math::vec4 aColor ) { mTextColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }

    void UIProgressBar::SetProgressValue( float aValue ) { mValue = aValue; }
    void UIProgressBar::SetProgressColor( math::vec4 aColor ) { mProgressColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }
    void UIProgressBar::SetThickness( float aValue ) { mThickness = aValue; }

    ImVec2 UIProgressBar::RequiredSize()
    {
        SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( mFont );
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );
        SE::Core::Engine::GetInstance()->UIContext()->PopFont();
        return lTextSize;
    }

    void UIProgressBar::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet ) ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        auto lTextSize     = ImGui::CalcTextSize( mText.c_str() );
        auto lTextPosition = GetContentAlignedposition( mHAlign, mVAlign, aPosition, lTextSize, aSize );

        ImGui::SetCursorPos( lTextPosition );
        ImGui::ProgressBar( mValue, ImVec2( 0.f, 0.f ), mText.c_str() );
        ImGui::SetCursorPos( aPosition );
        ImGui::Dummy( aSize );

        if( lTextColorSet ) ImGui::PopStyleColor();
    }

    // void *UIProgressBar::UIProgressBar_Create()
    // {
    //     auto lNewLabel = new UIProgressBar();

    //     return static_cast<void *>( lNewLabel );
    // }

    // void UIProgressBar::UIProgressBar_Destroy( void *aInstance ) { delete static_cast<UIProgressBar *>( aInstance ); }

    // void UIProgressBar::UIProgressBar_SetProgressValue( void *aInstance, float aValue )
    // {
    //     auto lInstance = static_cast<UIProgressBar *>( aInstance );

    //     lInstance->SetProgressValue( aValue );
    // }

    // void UIProgressBar::UIProgressBar_SetProgressColor( void *aInstance, math::vec4 aTextColor )
    // {
    //     auto lInstance = static_cast<UIProgressBar *>( aInstance );

    //     lInstance->SetProgressColor( aTextColor );
    // }

    // void UIProgressBar::UIProgressBar_SetText( void *aInstance, void *aText )
    // {
    //     auto lInstance = static_cast<UIProgressBar *>( aInstance );
    //     auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

    //     lInstance->SetText( lString );
    // }

    // void UIProgressBar::UIProgressBar_SetTextColor( void *aInstance, math::vec4 aTextColor )
    // {
    //     auto lInstance = static_cast<UIProgressBar *>( aInstance );

    //     lInstance->SetTextColor( aTextColor );
    // }

    // void UIProgressBar::UIProgressBar_SetThickness( void *aInstance, float aValue )
    // {
    //     auto lInstance = static_cast<UIProgressBar *>( aInstance );

    //     lInstance->SetThickness( aValue );
    // }

} // namespace SE::Core