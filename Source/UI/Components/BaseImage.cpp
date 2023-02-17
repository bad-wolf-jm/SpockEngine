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

    UIBaseImage::UIBaseImage( Ref<UIContext> aUIContext, Ref<VkSampler2D> aImage, math::vec2 aSize )
        : mImagePath{ "sampler://" }
        , mImage{ aImage }
        , mUIContext{ aUIContext }
        , mSize{ aSize.x, aSize.y }
    {

        mHandle = mUIContext->CreateTextureHandle( mImage );
    }

    void UIBaseImage::PushStyles() {}
    void UIBaseImage::PopStyles() {}

    void UIBaseImage::SetImage( fs::path const &aImagePath )
    {
        SE::Core::sTextureCreateInfo lTextureCreateInfo{};
        TextureData2D                lTextureData( lTextureCreateInfo, aImagePath );
        sTextureSamplingInfo         lSamplingInfo{};
        SE::Core::TextureSampler2D   lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

        auto lTexture = New<VkTexture2D>( mUIContext->GraphicContext(), lTextureData );
        mImage        = New<VkSampler2D>( mUIContext->GraphicContext(), lTexture, lSamplingInfo );
        mHandle       = mUIContext->CreateTextureHandle( mImage );
        mImagePath    = aImagePath;
    }

    void UIBaseImage::SetSize( float aWidth, float aHeight ) { mSize = ImVec2{ aWidth, aHeight }; }

    void UIBaseImage::SetRect( math::vec2 aTopLeft, math::vec2 aBottomRight )
    {
        mTopLeft     = ImVec2{ aTopLeft.x, aTopLeft.y };
        mBottomRight = ImVec2{ aBottomRight.x, aBottomRight.y };
    }

    void UIBaseImage::SetBackgroundColor( math::vec4 aColor ) { mBackgroundColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }

    void UIBaseImage::SetTintColor( math::vec4 aColor ) { mTintColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }

    ImVec2 UIBaseImage::RequiredSize() { return mSize; }

    void UIBaseImage::DrawContent( ImVec2 aPosition, ImVec2 aSize ) {}

} // namespace SE::Core