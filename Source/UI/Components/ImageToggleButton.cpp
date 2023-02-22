#include "ImageToggleButton.h"

namespace SE::Core
{
    UIImageToggleButton::UIImageToggleButton( std::function<void( bool )> aOnChange )
        : mOnChange{ aOnChange }
    {
    }

    void UIImageToggleButton::PushStyles() {}
    void UIImageToggleButton::PopStyles() {}

    void UIImageToggleButton::OnChange( std::function<void( bool )> aOnChange ) { mOnChange = aOnChange; }

    void UIImageToggleButton::SetActiveImage( UIBaseImage const& aImage ) { mActiveImage = aImage; }

    void UIImageToggleButton::SetInactiveImage( UIBaseImage const& aImage ) { mInactiveImage = aImage; }

    void UIImageToggleButton::PushStyles( bool aEnabled )
    {
        if( !aEnabled )
        {
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
        }
    }

    void UIImageToggleButton::PopStyles( bool aEnabled )
    {
        if( !aEnabled )
            ImGui::PopStyleColor( 4 );
    }

    ImVec2 UIImageToggleButton::RequiredSize()
    {
        PushStyles( mIsEnabled );

        auto lTextSize = ImGui::CalcTextSize( "m" );

        PopStyles( mIsEnabled );

        return lTextSize + ImGui::GetStyle().FramePadding * 2.0f;
    }

    void UIImageToggleButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        PushStyles( lEnabled );

        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, RequiredSize(), aSize ) );

        auto lImage = mActivated ? mActiveImage : mInactiveImage;

        if( ImGui::ImageButton( lImage.TextureID(), aSize ) && mOnChange && lEnabled )
        {
            mActivated = !mActivated;
            mOnChange( mActivated );
        }

        PopStyles( lEnabled );
    }

} // namespace SE::Core