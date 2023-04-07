#include "BaseImage.h"
#include "DotNet/Runtime.h"

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

    ImVec2 UIBaseImage::TopLeft() { return mTopLeft; }
    ImVec2 UIBaseImage::BottomRight() { return mBottomRight; }

    void   UIBaseImage::SetTintColor( math::vec4 aColor ) { mTintColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w }; }
    ImVec4 UIBaseImage::TintColor() { return mTintColor; }

    ImTextureID UIBaseImage::TextureID() { return static_cast<ImTextureID>( mHandle.Handle->GetVkDescriptorSet() ); }

    ImVec2 UIBaseImage::Size() { return mSize; }

    ImVec2 UIBaseImage::RequiredSize() { return mSize; }

    void UIBaseImage::DrawContent( ImVec2 aPosition, ImVec2 aSize ) {}

    void *UIBaseImage::UIBaseImage_Create()
    {
        auto lNewImage = new UIBaseImage();

        return static_cast<void *>( lNewImage );
    }

    void *UIBaseImage::UIBaseImage_CreateWithPath( void *aText, math::vec2 aSize )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewImage = new UIBaseImage( lString, aSize );

        return static_cast<void *>( lNewImage );
    }

    void UIBaseImage::UIBaseImage_Destroy( void *aInstance ) { delete static_cast<UIBaseImage *>( aInstance ); }

    void UIBaseImage::UIBaseImage_SetImage( void *aInstance, void *aPath )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aPath ) );

        lInstance->SetImage( lString );
    }

    void UIBaseImage::UIBaseImage_SetSize( void *aInstance, float aWidth, float aHeight )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );

        lInstance->SetSize( aWidth, aHeight );
    }

    void UIBaseImage::UIBaseImage_SetRect( void *aInstance, math::vec2 aTopLeft, math::vec2 aBottomRight )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );

        lInstance->SetRect( aTopLeft, aBottomRight );
    }

    // void UIBaseImage::UIBaseImage_SetBackgroundColor( void *aInstance, math::vec4 *aColor )
    // {
    //     auto lInstance = static_cast<UIBaseImage *>( aInstance );

    //     lInstance->SetBackgroundColor( *aColor );
    // }

    void UIBaseImage::UIBaseImage_SetTintColor( void *aInstance, math::vec4 aColor )
    {
        auto lInstance = static_cast<UIBaseImage *>( aInstance );

        lInstance->SetTintColor( aColor );
    }

} // namespace SE::Core