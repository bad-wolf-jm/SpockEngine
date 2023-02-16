#include "ImageButton.h"

namespace SE::Core
{
    UIImageButton::UIImageButton( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize,
                                  std::function<void()> aOnClick )
        : UIBaseImage( aUIContext, aImagePath, aSize )
        , mOnClick{ aOnClick }
    {
    }

    UIImageButton::UIImageButton( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize )
        : UIImageButton( aUIContext, aImagePath, aSize, std::function<void()>{} )
    {
    }

    void UIImageButton::PushStyles() {}
    void UIImageButton::PopStyles() {}

    UIImageButton &UIImageButton::OnClick( std::function<void()> aOnClick )
    {
        mOnClick = aOnClick;

        return *this;
    }

    ImVec2 UIImageButton::RequiredSize() { return mSize; }

    void UIImageButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        ImGui::SetCursorPos( aPosition );

        bool lClicked = ImGui::ImageButton( (ImTextureID)mHandle.Handle->GetVkDescriptorSet(), mSize, mTopLeft, mBottomRight, 0,
                                            mBackgroundColor, mTintColor );

        if( lClicked && mOnClick && lEnabled ) mOnClick();
    }

} // namespace SE::Core