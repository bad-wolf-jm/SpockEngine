#pragma once

#include "Core/GraphicContext//Texture2D.h"
#include "Core/Platform/EngineLoop.h"
#include "UI/UI.h"

#include <filesystem>

using namespace LTSE::Core;
using namespace LTSE::Core::UI;
using namespace LTSE::Graphics;

namespace LTSE::Editor
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

        Ref<LTSE::Graphics::Texture2D> m_DirectoryIcon;
        ImageHandle m_DirectoryIconHandle;

        Ref<LTSE::Graphics::Texture2D> m_FileIcon;
        ImageHandle m_FileIconHandle;

        Ref<LTSE::Graphics::Texture2D> m_BackIcon;
        ImageHandle m_BackIconHandle;

        float padding       = 5.0f;
        float thumbnailSize = 30.0f;
        float textSize      = 125.0f;
    };
} // namespace LTSE::Editor