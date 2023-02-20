#include "Image.h"

namespace SE::Core
{
    UIImage::UIImage( Ref<UIContext> aUIContext, fs::path const &aImagePath, math::vec2 aSize )
        : UIBaseImage(aUIContext, aImagePath, aSize)
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