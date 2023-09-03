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

        ContentBrowser( ref_t<IGraphicContext> aGraphicContext, ref_t<UIContext> aUIOverlay, fs::path aRoot );

        void Display();

      private:
        ref_t<IGraphicContext>  mGraphicContext;
        std::filesystem::path mRootDirectory;
        std::filesystem::path mCurrentDirectory;

        ref_t<SE::Graphics::ISampler2D> mDirectoryIcon;
        ImageHandle                   mDirectoryIconHandle;

        ref_t<SE::Graphics::ISampler2D> mFileIcon;
        ImageHandle                   mFileIconHandle;

        ref_t<SE::Graphics::ISampler2D> mBackIcon;
        ImageHandle                   mBackIconHandle;

        float mPadding       = 5.0f;
        float mThumbnailSize = 30.0f;
        float mTextSize      = 125.0f;
    };
} // namespace SE::Editor