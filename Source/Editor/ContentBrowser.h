#pragma once

// #include "Core/GraphicContext//ISampler2D.h"
#include "Engine/Engine.h"
#include "UI/UI.h"

#include <filesystem>

using namespace SE::Core;
using namespace SE::Core::UI;
using namespace SE::Graphics;

namespace SE::Editor
{
    class ContentBrowser
    {
      public:
        fs::path Root;

        ContentBrowser()  = default;
        ~ContentBrowser() = default;

        ContentBrowser( Ref<IGraphicContext> aGraphicContext, Ref<UIContext> aUIOverlay, fs::path aRoot );

        void Display();

      private:
        Ref<IGraphicContext>  mGraphicContext;
        std::filesystem::path mRootDirectory;
        std::filesystem::path mCurrentDirectory;

        Ref<SE::Graphics::ISampler2D> mDirectoryIcon;
        ImageHandle                   mDirectoryIconHandle;

        Ref<SE::Graphics::ISampler2D> mFileIcon;
        ImageHandle                   mFileIconHandle;

        Ref<SE::Graphics::ISampler2D> mBackIcon;
        ImageHandle                   mBackIconHandle;

        float mPadding       = 5.0f;
        float mThumbnailSize = 30.0f;
        float mTextSize      = 125.0f;
    };
} // namespace SE::Editor