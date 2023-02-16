#include "Image.h"

namespace SE::Core
{
    UIImage::UIImage( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize )
        : mImagePath{ aImagePath }
        , mUIContext{ aUIContext }
        , mSize{ aSize.x, aSize.y }
    {
        SetImage( aImagePath );
    }

    void UIImage::PushStyles() {}
    void UIImage::PopStyles() {}

    UIImage &UIImage::SetImage( fs::path const &aImagePath )
    {
        SE::Core::sTextureCreateInfo lTextureCreateInfo{};
        TextureData2D                lTextureData( lTextureCreateInfo, aImagePath );
        sTextureSamplingInfo         lSamplingInfo{};
        SE::Core::TextureSampler2D   lTextureSampler = SE::Core::TextureSampler2D( lTextureData, lSamplingInfo );

        auto lTexture = New<VkTexture2D>( mUIContext->GraphicContext(), lTextureData );
        mImage        = New<VkSampler2D>( mUIContext->GraphicContext(), lTexture, lSamplingInfo );
        mHandle       = mUIContext->CreateTextureHandle( mImage );
        mImagePath    = aImagePath;

        return *this;
    }

    UIImage &UIImage::SetSize( float aWidth, float aHeight )
    {
        mSize = ImVec2{ aWidth, aHeight };

        return *this;
    }

    UIImage &UIImage::SetBackgroundColor( math::vec4 aColor )
    {
        mBackgroundColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w };

        return *this;
    }

    UIImage &UIImage::SetTintColor( math::vec4 aColor )
    {
        mTintColor = ImVec4{ aColor.x, aColor.y, aColor.z, aColor.w };

        return *this;
    }

    ImVec2 UIImage::RequiredSize() { return mSize; }

    void UIImage::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( aPosition );

        ImGui::Image( (ImTextureID)mHandle.Handle->GetVkDescriptorSet(), aSize, ImVec2{ 0.0f, 0.0f }, ImVec2{ 1.0f, 1.0f },
                      mBackgroundColor, mTintColor );
    }

} // namespace SE::Core