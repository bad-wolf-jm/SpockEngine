#include "Plot.h"

namespace SE::Core
{

    UIPlot::UIPlot( std::string const &aText )
        : mText{ aText }
    {
    }

    void UIPlot::PushStyles() {}
    void UIPlot::PopStyles() {}

    void UIPlot::SetText( std::string const &aText ) { mText = aText; }

    ImVec2 UIPlot::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UIPlot::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( GetContentAlignedposition(mHAlign, mVAlign, aPosition, ImGui::CalcTextSize( mText.c_str() ), aSize) );

        ImGui::Text( mText.c_str(), aSize );
    }

} // namespace SE::Core