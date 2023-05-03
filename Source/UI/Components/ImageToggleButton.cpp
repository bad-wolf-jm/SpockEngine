#include "ImageToggleButton.h"
#include "DotNet/Runtime.h"
namespace SE::Core
{
    UIImageToggleButton::UIImageToggleButton( std::function<bool( bool )> aOnChange )
        : mOnClicked{ aOnChange }
    {
    }

    void UIImageToggleButton::PushStyles() {}
    void UIImageToggleButton::PopStyles() {}

    void UIImageToggleButton::OnClick( std::function<bool( bool )> aOnChange ) { mOnClicked = aOnChange; }
    void UIImageToggleButton::OnChanged( std::function<void()> aOnChanged ) { mOnChanged = aOnChanged; }

    void UIImageToggleButton::SetActiveImage( UIBaseImage *aImage ) { mActiveImage = aImage; }

    void UIImageToggleButton::SetInactiveImage( UIBaseImage *aImage ) { mInactiveImage = aImage; }

    bool UIImageToggleButton::IsActive() { return mActivated; }
    void UIImageToggleButton::SetActive( bool aValue ) { mActivated = aValue; }

    void UIImageToggleButton::PushStyles( bool aEnabled )
    {
        if( !aEnabled )
        {
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
        }
        else
        {
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 1.0f, 1.0f, 1.0f, 0.01f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 1.0f, 1.0f, 1.0f, 0.02f } );
        }
    }

    void UIImageToggleButton::PopStyles( bool aEnabled )
    {
        if( !aEnabled )
            ImGui::PopStyleColor( 4 );
        else
            ImGui::PopStyleColor( 3 );
    }

    ImVec2 UIImageToggleButton::RequiredSize()
    {
        PushStyles( mIsEnabled );

        auto lSize0 = mInactiveImage ? mInactiveImage->Size() : ImVec2{};
        auto lSize1 = mActiveImage ? mActiveImage->Size() : ImVec2{};

        auto lTextSize = ImVec2{ math::max( lSize0.x, lSize1.x ), math::max( lSize0.y, lSize1.y ) };

        PopStyles( mIsEnabled );

        return lTextSize + ImGui::GetStyle().FramePadding * 2.0f;
    }

    void UIImageToggleButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        PushStyles( lEnabled );

        auto lRequiredSize = RequiredSize();

        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, lRequiredSize, aSize ) );

        auto lImage = mActivated ? mActiveImage : mInactiveImage;

        if( lImage &&
            ImGui::ImageButton( lImage->TextureID(), lRequiredSize, lImage->TopLeft(), lImage->BottomRight(), 0,
                                lImage->BackgroundColor(), lImage->TintColor() ) &&
            mOnClicked && lEnabled )
            mActivated = mOnClicked( mActivated );

        PopStyles( lEnabled );
    }

    void *UIImageToggleButton::UIImageToggleButton_Create()
    {
        auto lNewImage = new UIImageToggleButton();

        return static_cast<void *>( lNewImage );
    }

    void UIImageToggleButton::UIImageToggleButton_Destroy( void *aInstance )
    {
        delete static_cast<UIImageToggleButton *>( aInstance );
    }

    bool UIImageToggleButton::UIImageToggleButton_IsActive( void *aInstance )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );

        return lInstance->IsActive();
    }

    void UIImageToggleButton::UIImageToggleButton_SetActive( void *aInstance, bool aValue )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );

        lInstance->SetActive( aValue );
    }

    void UIImageToggleButton::UIImageToggleButton_SetActiveImage( void *aInstance, void *aImage )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );
        auto lImage    = static_cast<UIBaseImage *>( aImage );

        lInstance->SetActiveImage( lImage );
    }

    void UIImageToggleButton::UIImageToggleButton_SetInactiveImage( void *aInstance, void *aImage )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );
        auto lImage    = static_cast<UIBaseImage *>( aImage );

        lInstance->SetInactiveImage( lImage );
    }

    void UIImageToggleButton::UIImageToggleButton_OnClicked( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnClickDelegate != nullptr ) mono_gchandle_free( lInstance->mOnClickDelegateHandle );

        lInstance->mOnClickDelegate       = aDelegate;
        lInstance->mOnClickDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnClick(
            [lInstance, lDelegate]( bool aValue )
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

                void *lParams[] = { (void *)&aValue };
                auto  lValue    = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );

                return *( (bool *)mono_object_unbox( lValue ) );
            } );
    }

    void UIImageToggleButton::UIImageToggleButton_OnChanged( void *aInstance, void *aDelegate )
    {
        auto lInstance = static_cast<UIImageToggleButton *>( aInstance );
        auto lDelegate = static_cast<MonoObject *>( aDelegate );

        if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

        lInstance->mOnChangeDelegate       = aDelegate;
        lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

        lInstance->OnChanged(
            [lInstance, lDelegate]()
            {
                auto lDelegateClass = mono_object_get_class( lDelegate );
                auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );
                auto lValue         = mono_runtime_invoke( lInvokeMethod, lDelegate, nullptr, nullptr );

                return *( (bool *)mono_object_unbox( lValue ) );
            } );
    }

} // namespace SE::Core