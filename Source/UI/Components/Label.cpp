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

    ImVec2 UILabel::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UILabel::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, ImGui::CalcTextSize( mText.c_str() ), aSize ) );

        ImGui::Text( mText.c_str(), aSize );
    }

} // namespace SE::Core