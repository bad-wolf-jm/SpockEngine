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

    void UIBaseImage::SetImage( fs::path const &aImagePath )
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

    // void *UIBaseImage::UIBaseImage_Create()
    // {
    //     auto lNewImage = new UIBaseImage();

    //     return static_cast<void *>( lNewImage );
    // }

    // void *UIBaseImage::UIBaseImage_CreateWithPath( void *aText, math::vec2 aSize )
    // {
    //     auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
    //     auto lNewImage = new UIBaseImage( lString, aSize );

    //     return static_cast<void *>( lNewImage );
    // }

    // void UIBaseImage::UIBaseImage_Destroy( void *aInstance ) { delete static_cast<UIBaseImage *>( aInstance ); }

    // void UIBaseImage::UIBaseImage_SetImage( void *aInstance, void *aPath )
    // {
    //     auto lInstance = static_cast<UIBaseImage *>( aInstance );
    //     auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aPath ) );

    //     lInstance->SetImage( lString );
    // }

    // void UIBaseImage::UIBaseImage_SetSize( void *aInstance, math::vec2 aSize )
    // {
    //     auto lInstance = static_cast<UIBaseImage *>( aInstance );

    //     lInstance->SetSize( aSize );
    // }

    // math::vec2 UIBaseImage::UIBaseImage_GetSize( void *aInstance )
    // {
    //     auto lInstance = static_cast<UIBaseImage *>( aInstance );
    //     auto lV        = lInstance->Size();

    //     return math::vec2{ lV.y, lV.y };
    // }

    // void UIBaseImage::UIBaseImage_SetTopLeft( void *aInstance, math::vec2 aTopLeft )
    // {
    //     auto lInstance = static_cast<UIBaseImage *>( aInstance );

    //     lInstance->SetTopLeft( aTopLeft );
    // }

    // math::vec2 UIBaseImage::UIBaseImage_GetTopLeft( void *aInstance )
    // {
    //     auto lInstance = static_cast<UIBaseImage *>( aInstance );
    //     auto lV        = lInstance->TopLeft();

    //     return math::vec2{ lV.y, lV.y };
    // }

    // void UIBaseImage::UIBaseImage_SetBottomRight( void *aInstance, math::vec2 aBottomRight )
    // {
    //     auto lInstance = static_cast<UIBaseImage *>( aInstance );

    //     lInstance->SetBottomRight( aBottomRight );
    // }

    // math::vec2 UIBaseImage::UIBaseImage_GetBottomRight( void *aInstance )
    // {
    //     auto lInstance = static_cast<UIBaseImage *>( aInstance );
    //     auto lV        = lInstance->BottomRight();

    //     return math::vec2{ lV.x, lV.y };
    // }

    // void UIBaseImage::UIBaseImage_SetTintColor( void *aInstance, math::vec4 aColor )
    // {
    //     auto lInstance = static_cast<UIBaseImage *>( aInstance );

    //     lInstance->SetTintColor( aColor );
    // }

    // math::vec4 UIBaseImage::UIBaseImage_GetTintColor( void *aInstance )
    // {
    //     auto lInstance = static_cast<UIBaseImage *>( aInstance );
    //     auto lV        = lInstance->TintColor();

    //     return math::vec4{ lV.x, lV.y, lV.z, lV.w };
    // }

} // namespace SE::Core