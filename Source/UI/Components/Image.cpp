#include "Image.h"
#include "DotNet/Runtime.h"

namespace SE::Core
{
    UIImage::UIImage( fs::path const &aImagePath, math::vec2 aSize )
        : UIBaseImage( aImagePath, aSize )
    {
    }

    UIImage::UIImage( Ref<ISampler2D> aImage, math::vec2 aSize )
        : UIBaseImage( aImage, aSize )
    {
    }

    ImVec2 UIImage::RequiredSize() { return mSize; }

    void UIImage::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, RequiredSize(), aSize ) );

        ImGui::Image( TextureID(), mSize, mTopLeft, mBottomRight, mTintColor, ImVec4{} );
    }

    // void *UIImage::UIImage_Create()
    // {
    //     auto lNewImage = new UIImage();

    //     return static_cast<void *>( lNewImage );
    // }

    // void *UIImage::UIImage_CreateWithPath( void *aText, math::vec2 aSize )
    // {
    //     auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
    //     auto lNewImage = new UIImage( lString, aSize );

    //     return static_cast<void *>( lNewImage );
    // }

    // void UIImage::UIImage_Destroy( void *aInstance ) { delete static_cast<UIImage *>( aInstance ); }

} // namespace SE::Core