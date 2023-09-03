#include "Form.h"

namespace SE::Core
{
    UIForm::UIForm( string_t const &aTitle )
        : mTitle{ aTitle }
    {
    }

    void UIForm::PushStyles()
    {
    }
    void UIForm::PopStyles()
    {
    }

    void UIForm::SetTitle( string_t const &aTitle )
    {
        mTitle = aTitle;
    }

    void UIForm::SetContent( UIComponent *aContent )
    {
        mContent = aContent;
    }

    void UIForm::SetSize( float aWidth, float aHeight )
    {
        mWidth         = aWidth;
        mHeight        = aHeight;
        mResizeRequest = true;
    }

    ImVec2 UIForm::RequiredSize()
    {
        if( mContent != nullptr )
            return mContent->RequiredSize();

        return ImVec2{ 100, 100 };
    }

    void UIForm::Update()
    {
        if( !mIsVisible )
            return;

        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2{ mPadding.z, mPadding.x } );
        
        if( mResizeRequest )
        {
            ImGui::SetNextWindowSize( ImVec2{ mWidth, mHeight }, ImGuiCond_Once );
            mResizeRequest = false;
        }

        if( ImGui::Begin( mTitle.c_str(), NULL, ImGuiWindowFlags_None ) )
        {
            ImVec2 lContentSize     = ImGui::GetContentRegionAvail();
            ImVec2 lContentPosition = ImGui::GetCursorPos();

            if( mContent != nullptr )
                mContent->Update( lContentPosition, lContentSize );
        }
        ImGui::End();
        ImGui::PopStyleVar();
    }

    void UIForm::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
    }

} // namespace SE::Core