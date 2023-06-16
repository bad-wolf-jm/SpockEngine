#include "BaseImage.h"

#include "Engine/Engine.h"

namespace SE::Core
{
    UIBaseImage::UIBaseImage( path_t const &aImagePath, math::vec2 aSize )
        : mImagePath{ aImagePath }
        , mSize{ aSize.x, aSize.y }
        , mTopLeft{ 0.0f, 0.0f }
        , mBottomRight{ 1.0f, 1.0f }

    {
        SetImage( aImagePath );
    }

    UIBaseImage::UIBaseImage( Ref<ISampler2D> aImage, math::vec2 aSize )
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

    void UIBaseImage::SetImage( path_t const &aImagePath )
    {
        SE::Core::sTextureCreateInfo lTextureCreateInfo{};
        TextureData2D                lTextureData( lTextureCreateInfo, aImagePath );
        sTextureSamplingInfo         lSamplingInfo{};
        SE::Core::TextureSampler2D   lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

        auto lTexture = CreateTexture2D( SE::Core::Engine::GetInstance()->GetGraphicContext(), lTextureData );
        mImage        = CreateSampler2D( SE::Core::Engine::GetInstance()->GetGraphicContext(), lTexture, lSamplingInfo );
        mHandle       = SE::Core::Engine::GetInstance()->UIContext()->CreateTextureHandle( mImage );
        mImagePath    = aImagePath;
    }

    ImVec2 UIBaseImage::Size() { return mSize; }
    void   UIBaseImage::SetSize( math::vec2 aSize ) { mSize = ImVec2{ aSize.x, aSize.y }; }
    void   UIBaseImage::SetSize( float aWidth, float aHeight ) { SetSize( math::vec2{ aWidth, aHeight } ); }

    ImVec2 UIBaseImage::TopLeft() { return mTopLeft; }
    void   UIBaseImage::SetTopLeft( math::vec2 aTopLeft ) { mTopLeft = ImVec2{ aTopLeft.x, aTopLeft.y }; }

    ImVec2 UIBaseImage::BottomRight() { return mBottomRight; }
    void   UIBaseImage::SetBottomRight( math::vec2 aBottomRight ) { mBottomRight = ImVec2{ aBottomRight.x, aBottomRight.y }; }

    ImVec4 UIBaseImage::TintColor() { return mTintColor; }
    void   UIBaseImage::SetTintColor( math::vec4 aColor ) { mTintColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }

    ImTextureID UIBaseImage::TextureID() { return static_cast<ImTextureID>( mHandle.Handle->GetID() ); }

    ImVec2 UIBaseImage::RequiredSize() { return mSize; }

    void UIBaseImage::DrawContent( ImVec2 aPosition, ImVec2 aSize ) {}
} // namespace SE::Core