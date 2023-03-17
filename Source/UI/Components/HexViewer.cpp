#include "HexViewer.h"

namespace SE::Core
{

    UIHexViewer::UIHexViewer( std::string const &aText )
        : mText{ aText }
    {
    }

    void UIHexViewer::PushStyles() {}
    void UIHexViewer::PopStyles() {}

    void UIHexViewer::SetText( std::string const &aText ) { mText = aText; }

    ImVec2 UIHexViewer::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UIHexViewer::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, ImGui::CalcTextSize( mText.c_str() ), aSize ) );

        ImGui::Text( mText.c_str(), aSize );
    }

} // namespace SE::Core