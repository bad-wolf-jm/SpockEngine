#include "ImageButton.h"

namespace SE::Core
{
    UIImageButton::UIImageButton( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize,
                                  std::function<void()> aOnClick )
        : mImagePath{ aImagePath }
        , mUIContext{ aUIContext }
        , mSize{ aSize.x, aSize.y }
        , mOnClick{ aOnClick }
    {
        SetImage( aImagePath );
    }

    UIImageButton::UIImageButton( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize )
        : UIImageButton( aUIContext, aImagePath, aSize, std::function<void()>{} )
    {
    }

    void UIImageButton::PushStyles() {}
    void UIImageButton::PopStyles() {}

    UIImageButton &UIImageButton::SetImage( fs::path const &aImagePath )
    {
        SE::Core::sTextureCreateInfo lTextureCreateInfo{};
        TextureData2D                lTextureData( lTextureCreateInfo, aImagePath );
        sTextureSamplingInfo         lSamplingInfo{};
        SE::Core::TextureSampler2D   lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

        auto lTexture = New<VkTexture2D>( mUIContext->GraphicContext(), lTextureData );
        mImage        = New<VkSampler2D>( mUIContext->GraphicContext(), lTexture, lSamplingInfo );
        mHandle       = mUIContext->CreateTextureHandle( mImage );
        mImagePath    = aImagePath;

        return *this;
    }

    UIImageButton &UIImageButton::SetSize( float aWidth, float aHeight )
    {
        mSize = ImVec2{ aWidth, aHeight };

        return *this;
    }

    UIImageButton &UIImageButton::SetBackgroundColor( math::vec4 aColor )
    {
        mBackgroundColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w };

        return *this;
    }

    UIImageButton &UIImageButton::SetTintColor( math::vec4 aColor )
    {
        mTintColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w };

        return *this;
    }

    ImVec2 UIImageButton::RequiredSize() { return mSize; }

    void UIImageButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        ImGui::SetCursorPos( aPosition );

        bool lClicked = ImGui::ImageButton( (ImTextureID)mHandle.Handle->GetVkDescriptorSet(), mSize, ImVec2{ 0.0f, 0.0f },
                                            ImVec2{ 1.0f, 1.0f }, 0, mBackgroundColor, mTintColor );

        if( lClicked && mOnClick && lEnabled ) mOnClick();
    }

} // namespace SE::Core