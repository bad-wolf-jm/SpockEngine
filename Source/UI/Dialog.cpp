#include "Dialog.h"

#include "DotNet/Runtime.h"
namespace SE::Core
{
    UIDialog::UIDialog()
    {
        mIsVisible = false;
    }

    UIDialog::UIDialog( string_t aTitle, math::vec2 aSize )
        : mTitle{ aTitle }
        , mSize{ aSize }
    {
        mIsVisible = false;
    }

    void UIDialog::PushStyles()
    {
    }
    void UIDialog::PopStyles()
    {
    }

    void UIDialog::SetTitle( string_t const &aTitle )
    {
        mTitle = aTitle;
    }
    void UIDialog::SetSize( math::vec2 aSize )
    {
        mSize = aSize;
    }
    void UIDialog::SetContent( UIComponent *aContent )
    {
        mContent = aContent;
    }

    void UIDialog::Open()
    {
        mIsVisible = true;
        if( !ImGui::IsPopupOpen( mTitle.c_str() ) )
            ImGui::OpenPopup( mTitle.c_str() );
    }

    void UIDialog::Close()
    {
        ImGui::CloseCurrentPopup();
        mIsVisible = false;
    }

    void UIDialog::Update()
    {
        if( !mIsVisible )
            return;

        if( !ImGui::IsPopupOpen( mTitle.c_str() ) )
            ImGui::OpenPopup( mTitle.c_str() );
        ImGui::SetNextWindowSize( ImVec2{ mSize.x, mSize.y } );
        ImGuiWindowFlags lFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoResize |
                                  ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2{ mPadding.z, mPadding.x } );

        bool lOpen = true;
        if( ImGui::BeginPopupModal( mTitle.c_str(), &lOpen, lFlags ) )
        {
            ImVec2 lContentSize     = ImGui::GetContentRegionAvail();
            ImVec2 lContentPosition = ImGui::GetCursorPos();

            if( mContent != nullptr )
                mContent->Update( lContentPosition, lContentSize );

            ImGui::EndPopup();
        }
        if( !lOpen )
            Close();
        ImGui::PopStyleVar();
    }

    ImVec2 UIDialog::RequiredSize()
    {
        if( mContent != nullptr )
            return mContent->RequiredSize();

        return ImVec2{ 100, 100 };
    }

    void UIDialog::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
    }

    void *UIDialog::UIDialog_Create()
    {
        auto lNewDialog = new UIDialog();

        return static_cast<void *>( lNewDialog );
    }

    void *UIDialog::UIDialog_CreateWithTitleAndSize( void *aTitle, math::vec2 *aSize )
    {
        auto lString    = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );
        auto lNewDialog = new UIDialog( lString, *aSize );

        return static_cast<void *>( lNewDialog );
    }

    void UIDialog::UIDialog_Destroy( void *aInstance )
    {
        delete static_cast<UIDialog *>( aInstance );
    }

    void UIDialog::UIDialog_SetTitle( void *aInstance, void *aTitle )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );
        auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );

        lInstance->SetTitle( lString );
    }

    void UIDialog::UIDialog_SetSize( void *aInstance, math::vec2 aSize )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );

        lInstance->SetSize( aSize );
    }

    void UIDialog::UIDialog_SetContent( void *aInstance, void *aContent )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );
        auto lContent  = static_cast<UIComponent *>( aContent );

        lInstance->SetContent( lContent );
    }

    void UIDialog::UIDialog_Open( void *aInstance )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );

        lInstance->Open();
    }

    void UIDialog::UIDialog_Close( void *aInstance )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );

        lInstance->Close();
    }

    void UIDialog::UIDialog_Update( void *aInstance )
    {
        auto lInstance = static_cast<UIDialog *>( aInstance );

        lInstance->Update();
    }
} // namespace SE::Core