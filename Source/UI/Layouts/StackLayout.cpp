#include "StackLayout.h"

namespace SE::Core
{
    void UIStackLayout::PushStyles() {}
    void UIStackLayout::PopStyles() {}

    ImVec2 UIStackLayout::RequiredSize()
    {
        float lWidth  = 0.0f;
        float lHeight = 0.0f;

        for( auto const &[lKey, lValue] : mChildren )
        {
            ImVec2 lRequiredSize{};
            if( lValue ) lRequiredSize = lValue->RequiredSize();
            lWidth  = math::max( lWidth, lRequiredSize.x );
            lHeight = math::max( lHeight, lRequiredSize.y );
        }

        return ImVec2{ lWidth, lHeight };
    }

    void UIStackLayout::Add( UIComponent *aChild, string_t const &aKey )
    {
        mChildren[aKey] = aChild;
        if( mCurrent.empty() ) mCurrent = aKey;
    }

    void UIStackLayout::SetCurrent( string_t const &aKey )
    {
        if( mChildren.find( aKey ) != mChildren.end() )
            mCurrent = aKey;
        else
            mCurrent = "";
    }

    void UIStackLayout::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        if( mCurrent.empty() ) return;
        if( mChildren[mCurrent] == nullptr ) return;

        ImGui::SetCursorPos( aPosition );
        ImGui::PushID( (void *)mChildren[mCurrent] );
        ImGui::BeginChild( "##LayoutItem", aSize );
        mChildren[mCurrent]->Update( ImVec2{}, aSize );
        ImGui::EndChild();
        ImGui::PopID();
    }
} // namespace SE::Core