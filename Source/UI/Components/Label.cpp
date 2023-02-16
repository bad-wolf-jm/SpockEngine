#include "Label.h"

namespace SE::Core
{
    UILabel::UILabel( std::string const &aText )
        : mText{ aText }
    {
    }

    void UILabel::PushStyles() {}
    void UILabel::PopStyles() {}

    UILabel &UILabel::SetText( std::string const &aText )
    {
        mText = aText;
        
        return *this;
    }

    ImVec2 UILabel::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UILabel::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( aPosition );

        ImGui::Text( mText.c_str(), aSize );
    }

} // namespace SE::Core