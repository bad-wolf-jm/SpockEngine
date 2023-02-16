#pragma once

#include "Component.h"
#include "Core/GraphicContext/UI/UIContext.h"

namespace SE::Core
{
    class UIImageButton : public UIComponent
    {
      public:
        UIImageButton() = default;

        UIImageButton(  Ref<UIContext> aUIContext, fs::path const &aImagePath );
        UIImageButton(  Ref<UIContext> aUIContext, fs::path const &aImagePath, std::function<void()> aOnClick );

      private:
        Ref<UIContext>        mUIContext;
        fs::path              mImagePath;
        std::function<void()> mOnClick;

        Ref<VkSampler2D> mImage;
        ImageHandle      mImageHandle;

      private:
        void PushStyles();
        void PopStyles();

        void PushStyles( bool aEnabled );
        void PopStyles( bool aEnabled );

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core