#include "ImageButton.h"

namespace SE::Core
{
    UIImageButton::UIImageButton( Ref<UIContext> aUIContext, fs::path const &aImagePath )
        : mImagePath{ aImagePath }
        , mUIContext{ aUIContext }

    {
        SE::Core::sTextureCreateInfo lTextureCreateInfo{};
        TextureData2D                lTextureData( lTextureCreateInfo, aImagePath );
        sTextureSamplingInfo         lSamplingInfo{};
        SE::Core::TextureSampler2D   lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

        auto lTexture = New<VkTexture2D>( mUIContext->GraphicContext(), lTextureData );
        mImage        = New<VkSampler2D>( mUIContext->GraphicContext(), lTexture, lSamplingInfo );
        mImageHandle  = mUIContext->CreateTextureHandle( mImage );
    }

    UIImageButton::UIImageButton( Ref<UIContext> aUIContext, fs::path const &aImagePath, std::function<void()> aOnClick )
        : mImagePath{ aImagePath }
        , mUIContext{ aUIContext }
        , mOnClick{ aOnClick }
    {
        SE::Core::sTextureCreateInfo lTextureCreateInfo{};
        TextureData2D                lTextureData( lTextureCreateInfo, aImagePath );
        sTextureSamplingInfo         lSamplingInfo{};
        SE::Core::TextureSampler2D   lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

        auto lTexture = New<VkTexture2D>( mUIContext->GraphicContext(), lTextureData );
        mImage        = New<VkSampler2D>( mUIContext->GraphicContext(), lTexture, lSamplingInfo );
        mImageHandle  = mUIContext->CreateTextureHandle( mImage );
    }

    void UIImageButton::PushStyles() {}
    void UIImageButton::PopStyles() {}

    void UIImageButton::PushStyles( bool aEnabled )
    {
        if( !aEnabled )
        {
            ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.3f, 0.3f, 0.3f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.1f, 0.1f, .2f } );
        }
    }

    void UIImageButton::PopStyles( bool aEnabled )
    {
        if( !aEnabled ) ImGui::PopStyleColor( 4 );
    }

    ImVec2 UIImageButton::RequiredSize()
    {
        // PushStyles( mIsEnabled );

        // auto lTextSize = ImGui::CalcTextSize( mText.c_str() );

        // PopStyles( mIsEnabled );

        // return lTextSize + ImGui::GetStyle().FramePadding * 2.0f;
        return ImVec2{};
    }

    void UIImageButton::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        bool lEnabled = mIsEnabled;

        PushStyles( lEnabled );

        ImGui::SetCursorPos( aPosition );

        bool lClicked =
            ImGui::ImageButton( (ImTextureID)mImageHandle.Handle->GetVkDescriptorSet(), ImVec2{ 22.0f, 22.0f }, ImVec2{ 0.0f, 0.0f },
                                ImVec2{ 1.0f, 1.0f }, 0, ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f }, ImVec4{ 0.0f, 1.0f, 0.0f, 0.8f } );
        if( lClicked && mOnClick && lEnabled ) mOnClick();

        PopStyles( lEnabled );
    }

} // namespace SE::Core