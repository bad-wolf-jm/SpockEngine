#include "Label.h"

namespace SE::Core
{

    UILabel::UILabel( std::string const &aText )
        : mText{ aText }
    {
    }

    void UILabel::PushStyles() {}
    void UILabel::PopStyles() {}

    void UILabel::SetText( std::string const &aText ) { mText = aText; }
    void UILabel::SetTextColor( math::vec4 aColor ) { mTextColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }

    ImVec2 UILabel::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UILabel::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet ) ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );
        auto lTextPosition = GetContentAlignedposition( mHAlign, mVAlign, aPosition, lTextSize, aSize );

        ImGui::SetCursorPos( lTextPosition );
        ImGui::Text( mText.c_str(), aSize );
        ImGui::SetCursorPos( aPosition );
        ImGui::Dummy(aSize);

        if( lTextColorSet ) ImGui::PopStyleColor();
    }

} // namespace SE::Core