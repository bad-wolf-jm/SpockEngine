#include "BaseImage.h"

namespace SE::Core
{
    UIBaseImage::UIBaseImage( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize )
        : mImagePath{ aImagePath }
        , mUIContext{ aUIContext }
        , mSize{ aSize.x, aSize.y }
    {
        SetImage( aImagePath );
    }

    void UIBaseImage::PushStyles() {}
    void UIBaseImage::PopStyles() {}

    UIBaseImage &UIBaseImage::SetImage( fs::path const &aImagePath )
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

    UIBaseImage &UIBaseImage::SetSize( float aWidth, float aHeight )
    {
        mSize = ImVec2{ aWidth, aHeight };

        return *this;
    }

    UIBaseImage &UIBaseImage::SetRect( math::vec2 aTopLeft, math::vec2 aBottomRight )
    {
        mTopLeft = ImVec2{ aTopLeft.x, aTopLeft.y };
        mBottomRight = ImVec2{ aBottomRight.x, aBottomRight.y };

        return *this;
    }

    UIBaseImage &UIBaseImage::SetBackgroundColor( math::vec4 aColor )
    {
        mBackgroundColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w };

        return *this;
    }

    UIBaseImage &UIBaseImage::SetTintColor( math::vec4 aColor )
    {
        mTintColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w };

        return *this;
    }

    ImVec2 UIBaseImage::RequiredSize() { return mSize; }

    void UIBaseImage::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
    }

} // namespace SE::Core