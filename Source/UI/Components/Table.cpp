#include "Table.h"

namespace SE::Core
{

    UITable::UITable( std::string const &aText )
        : mText{ aText }
    {
    }

    void UITable::PushStyles() {}
    void UITable::PopStyles() {}

    void UITable::SetText( std::string const &aText ) { mText = aText; }

    ImVec2 UITable::RequiredSize()
    {
        auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        return lTextSize;
    }

    void UITable::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( GetContentAlignedposition(mHAlign, mVAlign, aPosition, ImGui::CalcTextSize( mText.c_str() ), aSize) );

        ImGui::Text( mText.c_str(), aSize );
    }

} // namespace SE::Core