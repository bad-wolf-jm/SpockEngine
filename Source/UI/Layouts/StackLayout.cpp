#include "StackLayout.h"
#include "DotNet/Runtime.h"

namespace SE::Core
{
    void StackLayout::PushStyles() {}
    void StackLayout::PopStyles() {}

    ImVec2 StackLayout::RequiredSize()
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

    void StackLayout::Add( UIComponent *aChild, std::string const &aKey )
    {
        mChildren[aKey] = aChild;
        if( mCurrent.empty() ) mCurrent = aKey;
    }

    void StackLayout::SetCurrent( std::string const &aKey )
    {
        if( mChildren.find( aKey ) != mChildren.end() )
            mCurrent = aKey;
        else
            mCurrent = "";
    }

    void StackLayout::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        if( mCurrent.empty() ) return;

        ImGui::SetCursorPos( aPosition );
        ImGui::PushID( (void *)mChildren[mCurrent] );
        ImGui::BeginChild( "##LayoutItem", aSize );
        mChildren[mCurrent]->Update( ImVec2{}, aSize );
        ImGui::EndChild();
        ImGui::PopID();
    }

    void *StackLayout::StackLayout_Create()
    {
        auto lNewLayout = new StackLayout();

        return static_cast<void *>( lNewLayout );
    }

    void StackLayout::StackLayout_Destroy( void *aInstance ) { delete static_cast<StackLayout *>( aInstance ); }

    void StackLayout::StackLayout_Add( void *aInstance, void *aChild, void *aKey )
    {
        auto lInstance = static_cast<StackLayout *>( aInstance );
        auto lChild    = static_cast<UIComponent *>( aChild );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aKey ) );

        lInstance->Add( lChild, lString );
    }

    void StackLayout::StackLayout_SetCurrent( void *aInstance, void *aKey )
    {
        auto lInstance = static_cast<StackLayout *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aKey ) );

        lInstance->SetCurrent( lString );
    }
} // namespace SE::Core