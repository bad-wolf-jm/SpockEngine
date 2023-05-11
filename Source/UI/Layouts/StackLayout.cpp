#include "StackLayout.h"
#include "DotNet/Runtime.h"

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

    void UIStackLayout::Add( UIComponent *aChild, std::string const &aKey )
    {
        mChildren[aKey] = aChild;
        if( mCurrent.empty() ) mCurrent = aKey;
    }

    void UIStackLayout::SetCurrent( std::string const &aKey )
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

    void *UIStackLayout::UIStackLayout_Create()
    {
        auto lNewLayout = new UIStackLayout();

        return static_cast<void *>( lNewLayout );
    }

    void UIStackLayout::UIStackLayout_Destroy( void *aInstance ) { delete static_cast<UIStackLayout *>( aInstance ); }

    void UIStackLayout::UIStackLayout_Add( void *aInstance, void *aChild, void *aKey )
    {
        auto lInstance = static_cast<UIStackLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aKey ) );

        lInstance->Add( lChild, lString );
    }

    void UIStackLayout::UIStackLayout_SetCurrent( void *aInstance, void *aKey )
    {
        auto lInstance = static_cast<UIStackLayout *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aKey ) );

        lInstance->SetCurrent( lString );
    }
} // namespace SE::Core