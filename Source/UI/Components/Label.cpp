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
    void UILabel::SetBackgroundColor( math::vec4 aColor ) { mBackgroundColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }

    ImVec2 UILabel::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UILabel::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, ImGui::CalcTextSize( mText.c_str() ), aSize ) );
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) && ( mTextColor.y != 0.0f ) && ( mTextColor.z != 0.0f ) && ( mTextColor.w != 0.0f ) );

        if( ( mBackgroundColor.x != 0.0f ) && ( mBackgroundColor.y != 0.0f ) && ( mBackgroundColor.z != 0.0f ) &&
            ( mBackgroundColor.w != 0.0f ) )
        {
            auto lDrawList       = ImGui::GetWindowDrawList();
            auto lScreenPosition = ImGui::GetCursorScreenPos();
            lDrawList->AddRectFilled( lScreenPosition, lScreenPosition + aSize, ImColor(mBackgroundColor), 0.0f );
        }

        if( lTextColorSet ) ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );
        ImGui::Text( mText.c_str(), aSize );
        if( lTextColorSet ) ImGui::PopStyleColor();
    }

} // namespace SE::Core