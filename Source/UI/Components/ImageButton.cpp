#include "ImageButton.h"

namespace SE::Core
{
    UIImageButton::UIImageButton( fs::path const &aImagePath, math::vec2 aSize,
                                  std::function<void()> aOnClick )
        : UIBaseImage( aImagePath, aSize )
        , mOnClick{ aOnClick }
    {
    }

    UIImageButton::UIImageButton( fs::path const &aImagePath, math::vec2 aSize )
        : UIImageButton( aImagePath, aSize, std::function<void()>{} )
    {
    }

    UIImageButton::UIImageButton( Ref<VkSampler2D> aImage, math::vec2 aSize )
        : UIImageButton( aImage, aSize, std::function<void()>{} )
    {
    }

    UIImageButton::UIImageButton( Ref<VkSampler2D> aImage, math::vec2 aSize, std::function<void()> aOnClick )
        : UIBaseImage( aImage, aSize )
        , mOnClick{ aOnClick }
    {
    }

    void UIImageButton::PushStyles() {}
    void UIImageButton::PopStyles() {}

    void UIImageButton::OnClick( std::function<void()> aOnClick )
    {
        mOnClick = aOnClick;
    }

    ImVec2 UIImageButton::RequiredSize() { return mSize; }

    void UIImageButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, RequiredSize(), aSize ) );

        bool lClicked = ImGui::ImageButton( TextureID(), mSize, mTopLeft, mBottomRight, 0,
                                            mBackgroundColor, mTintColor );

        if( lClicked && mOnClick && lEnabled ) mOnClick();
    }

} // namespace SE::Core