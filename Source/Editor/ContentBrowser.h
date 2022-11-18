#pragma once

#include "Core/GraphicContext//Texture2D.h"
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

        ContentBrowser( GraphicContext &aGraphicContext, Ref<UIContext> aUIOverlay, fs::path aRoot );

        void Display();

      private:
        GraphicContext mGraphicContext;
        std::filesystem::path m_RootDirectory;
        std::filesystem::path m_CurrentDirectory;

        Ref<SE::Graphics::Texture2D> m_DirectoryIcon;
        ImageHandle m_DirectoryIconHandle;

        Ref<SE::Graphics::Texture2D> m_FileIcon;
        ImageHandle m_FileIconHandle;

        Ref<SE::Graphics::Texture2D> m_BackIcon;
        ImageHandle m_BackIconHandle;

        float padding       = 5.0f;
        float thumbnailSize = 30.0f;
        float textSize      = 125.0f;
    };
} // namespace SE::Editor