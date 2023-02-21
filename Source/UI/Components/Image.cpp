#include "Image.h"

namespace SE::Core
{
    UIImage::UIImage( fs::path const &aImagePath, math::vec2 aSize )
        : UIBaseImage(aImagePath, aSize)
    {
    }

    UIImage::UIImage( Ref<VkSampler2D> aImage, math::vec2 aSize )
        : UIBaseImage(aImage, aSize)
    {
    }

    ImVec2 UIImage::RequiredSize() { return mSize; }

    void UIImage::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        ImGui::SetCursorPos( GetContentAlignedposition( mHAlign, mVAlign, aPosition, RequiredSize(), aSize ) );

        ImGui::Image( (ImTextureID)mHandle.Handle->GetVkDescriptorSet(), mSize, mTopLeft, mBottomRight,
                      mTintColor, ImVec4{} );
    }

} // namespace SE::Core