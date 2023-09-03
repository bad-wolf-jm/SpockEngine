#include "ImageButton.h"
#include "DotNet/Runtime.h"

namespace SE::Core
{
    UIImageButton::UIImageButton( fs::path const &aImagePath, math::vec2 aSize, std::function<void()> aOnClick )
        : UIBaseImage( aImagePath, aSize )
        , mOnClick{ aOnClick }
    {
    }

    UIImageButton::UIImageButton( fs::path const &aImagePath, math::vec2 aSize )
        : UIImageButton( aImagePath, aSize, std::function<void()>{} )
    {
    }

    UIImageButton::UIImageButton( ref_t<ISampler2D> aImage, math::vec2 aSize )
        : UIImageButton( aImage, aSize, std::function<void()>{} )
    {
    }

    UIImageButton::UIImageButton( ref_t<ISampler2D> aImage, math::vec2 aSize, std::function<void()> aOnClick )
        : UIBaseImage( aImage, aSize )
        , mOnClick{ aOnClick }
    {
    }

    void UIImageButton::PushStyles()
    {
    }
    void UIImageButton::PopStyles()
    {
    }

    void UIImageButton::OnClick( std::function<void()> aOnClick )
    {
        mOnClick = aOnClick;
    }

    ImVec2 UIImageButton::RequiredSize()
    {
        return mSize;
    }

    void UIImageButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, RequiredSize(), aSize ) );

        bool lClicked = ImGui::ImageButton( TextureID(), mSize, mTopLeft, mBottomRight, 0, mBackgroundColor, mTintColor );

        if( lClicked && mOnClick && lEnabled )
            mOnClick();
    }

    void *UIImageButton::UIImageButton_Create()
    {
        auto lNewImage = new UIImageButton();

        return static_cast<void *>( lNewImage );
    }

    void *UIImageButton::UIImageButton_CreateWithPath( void *aText, math::vec2 *aSize )
    {
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );
        auto lNewImage = new UIImageButton( lString, *aSize );

        return static_cast<void *>( lNewImage );
    }

    void UIImageButton::UIImageButton_Destroy( void *aInstance )
    {
        delete static_cast<UIImageButton *>( aInstance );
    }

    void UIImageButton::UIImageButton_OnClick( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageButton *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnClickDelegate != nullptr )
            mono_gchandle_free( lInstance->mOnClickDelegateHandle );

        lInstance->mOnClickDelegate       = aDelegate;
        lInstance->mOnClickDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnClick(
            [lInstance, lDelegate]()
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );
            } );
    }

} // namespace SE::Core