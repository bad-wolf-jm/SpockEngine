#include "Label.h"
#include "Engine/Engine.h"

#include "DotNet/Runtime.h"

namespace SE::Core
{
    UILabel::UILabel( std::string const &aText )
        : mText{ aText }
    {
    }

    void UILabel::PushStyles()
    {
    }
    void UILabel::PopStyles()
    {
    }

    void UILabel::SetText( std::string const &aText )
    {
        mText = aText;
    }
    void UILabel::SetTextColor( math::vec4 aColor )
    {
        mTextColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w };
    }

    ImVec2 UILabel::RequiredSize()
    {
        SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( mFont );
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );
        SE::Core::Engine::GetInstance()->UIContext()->PopFont();

        return lTextSize + UIComponent::RequiredSize();
    }

    void UILabel::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet )
            ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        auto lTextSize     = ImGui::CalcTextSize( mText.c_str() );
        auto lTextPosition = GetContentAlignedposition( mHAlign, mVAlign, aPosition, lTextSize, aSize );

        ImGui::SetCursorPos( lTextPosition );
        ImGui::Text( mText.c_str(), aSize );
        ImGui::SetCursorPos( aPosition );
        ImGui::Dummy( aSize );

        if( lTextColorSet )
            ImGui::PopStyleColor();
    }

    void *UILabel::UILabel_Create()
    {
        auto lNewLabel = new UILabel();

        return static_cast<void *>( lNewLabel );
    }

    void *UILabel::UILabel_CreateWithText( void *aText )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewLabel = new UILabel( lString );

        return static_cast<void *>( lNewLabel );
    }

    void UILabel::UILabel_Destroy( void *aInstance )
    {
        delete static_cast<UILabel *>( aInstance );
    }

    void UILabel::UILabel_SetText( void *aInstance, void *aText )
    {
        auto lInstance = static_cast<UILabel *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lInstance->SetText( lString );
    }

    void UILabel::UILabel_SetTextColor( void *aInstance, math::vec4 aTextColor )
    {
        auto lInstance = static_cast<UILabel *>( aInstance );

        lInstance->SetTextColor( aTextColor );
    }
} // namespace SE::Core