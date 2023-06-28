#include "Markdown.h"
#include "Engine/Engine.h"



namespace SE::Core
{
    UIMarkdown::UIMarkdown( string_t const &aText )
        : mText{ aText }
    {
    }

    void UIMarkdown::PushStyles() {}
    void UIMarkdown::PopStyles() {}

    void UIMarkdown::SetText( string_t const &aText ) { mText = aText; }
    void UIMarkdown::SetTextColor( math::vec4 aColor ) { mTextColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }

    ImVec2 UIMarkdown::RequiredSize()
    {
        return ImVec2{0,0};
    }

    void UIMarkdown::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lTextColorSet =
            ( ( mTextColor.x != 0.0f ) || ( mTextColor.y != 0.0f ) || ( mTextColor.z != 0.0f ) || ( mTextColor.w != 0.0f ) );
        if( lTextColorSet ) ImGui::PushStyleColor( ImGuiCol_Text, mTextColor );

        ImGui::SetCursorPos( aPosition );
        mRenderer.print(mText.c_str(), mText.c_str() + mText.size());

        if( lTextColorSet ) ImGui::PopStyleColor();
    }
} // namespace SE::Core