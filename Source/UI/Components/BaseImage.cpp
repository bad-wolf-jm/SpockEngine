#include "BaseImage.h"

#include "Engine/Engine.h"

namespace SE::Core
{
    UIBaseImage::UIBaseImage( fs::path const &aImagePath, math::vec2 aSize )
        : mImagePath{ aImagePath }
        , mSize{ aSize.x, aSize.y }
        , mTopLeft{ 0.0f, 0.0f }
        , mBottomRight{ 1.0f, 1.0f }

    {
        SetImage( aImagePath );
    }

    UIBaseImage::UIBaseImage( Ref<VkSampler2D> aImage, math::vec2 aSize )
        : mImagePath{ "sampler://" }
        , mImage{ aImage }
        , mSize{ aSize.x, aSize.y }
        , mTopLeft{ 0.0f, 0.0f }
        , mBottomRight{ 1.0f, 1.0f }
    {
        mHandle = SE::Core::Engine::GetInstance()->UIContext()->CreateTextureHandle( mImage );
    }

    void UIBaseImage::PushStyles() {}
    void UIBaseImage::PopStyles() {}

    void UIBaseImage::SetImage( fs::path const &aImagePath )
    {
        SE::Core::sTextureCreateInfo lTextureCreateInfo{};
        TextureData2D                lTextureData( lTextureCreateInfo, aImagePath );
        sTextureSamplingInfo         lSamplingInfo{};
        SE::Core::TextureSampler2D   lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

        auto lTexture = New<VkTexture2D>( SE::Core::Engine::GetInstance()->GetGraphicContext(), lTextureData );
        mImage        = New<VkSampler2D>( SE::Core::Engine::GetInstance()->GetGraphicContext(), lTexture, lSamplingInfo );
        mHandle       = SE::Core::Engine::GetInstance()->UIContext()->CreateTextureHandle( mImage );
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