#include "ImageToggleButton.h"
#include "DotNet/Runtime.h"
namespace SE::Core
{
    UIImageToggleButton::UIImageToggleButton( std::function<bool( bool )> aOnChange )
        : mOnClicked{ aOnChange }
    {
    }

    void UIImageToggleButton::PushStyles() {}
    void UIImageToggleButton::PopStyles() {}

    void UIImageToggleButton::OnClick( std::function<bool( bool )> aOnChange ) { mOnClicked = aOnChange; }
    void UIImageToggleButton::OnChanged( std::function<void()> aOnChanged ) { mOnChanged = aOnChanged; }

    void UIImageToggleButton::SetActiveImage( UIBaseImage *aImage ) { mActiveImage = aImage; }

    void UIImageToggleButton::SetInactiveImage( UIBaseImage *aImage ) { mInactiveImage = aImage; }

    bool UIImageToggleButton::IsActive() { return mActivated; }
    void UIImageToggleButton::SetActive( bool aValue ) { mActivated = aValue; }

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

        auto lSize0 = mInactiveImage ? mInactiveImage->Size() : ImVec2{};
        auto lSize1 = mActiveImage ? mActiveImage->Size() : ImVec2{};

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

        if( lImage &&
            ImGui::ImageButton( lImage->TextureID(), lRequiredSize, lImage->TopLeft(), lImage->BottomRight(), 0,
                                lImage->BackgroundColor(), lImage->TintColor() ) &&
            mOnClicked && lEnabled )
            mActivated = mOnClicked( mActivated );

        PopStyles( lEnabled );
    }

} // namespace SE::Core