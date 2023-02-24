#include "ImageToggleButton.h"

namespace SE::Core
{
    UIImageToggleButton::UIImageToggleButton( std::function<bool( bool )> aOnChange )
        : mOnChange{ aOnChange }
    {
    }

    void UIImageToggleButton::PushStyles() {}
    void UIImageToggleButton::PopStyles() {}

    void UIImageToggleButton::OnChange( std::function<bool( bool )> aOnChange ) { mOnChange = aOnChange; }

    void UIImageToggleButton::SetActiveImage( UIBaseImage const &aImage ) { mActiveImage = aImage; }

    void UIImageToggleButton::SetInactiveImage( UIBaseImage const &aImage ) { mInactiveImage = aImage; }

    void UIImageToggleButton::PushStyles( bool aEnabled )
    {
        if( !aEnabled )
        {
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
        }
        else
        {
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0f, 1.0f, 1.0f, 0.01f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0f, 1.0f, 1.0f, 0.02f } );
        }
    }

    void UIImageToggleButton::PopStyles( bool aEnabled )
    {
        if( !aEnabled )
            ImGui::PopStyleColor( 4 );
        else
            ImGui::PopStyleColor( 3 );
    }

    ImVec2 UIImageToggleButton::RequiredSize()
    {
        PushStyles( mIsEnabled );

        auto lSize0 = mInactiveImage.Size();
        auto lSize1 = mActiveImage.Size();

        auto lTextSize = ImVec2{ math::max( lSize0.x, lSize1.x ), math::max( lSize0.y, lSize1.y ) };

        PopStyles( mIsEnabled );

        return lTextSize + ImGui::GetStyle().FramePadding * 2.0f;
    }

    void UIImageToggleButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        PushStyles( lEnabled );

        auto lRequiredSize = RequiredSize();

        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, lRequiredSize, aSize ) );

        auto lImage = mActivated ? mActiveImage : mInactiveImage;

        if( ImGui::ImageButton( lImage.TextureID(), lRequiredSize, lImage.TopLeft(), lImage.BottomRight(), 0, lImage.BackgroundColor(),
                                lImage.TintColor() ) &&
            mOnChange && lEnabled )
            mActivated = mOnChange( mActivated );

        PopStyles( lEnabled );
    }

} // namespace SE::Core