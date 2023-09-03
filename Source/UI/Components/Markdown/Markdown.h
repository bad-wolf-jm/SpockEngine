#pragma once

#include "MarkdownRenderer.h"
#include "UI/Components/Component.h"

namespace SE::Core
{
    struct UIMarkdownRenderer : public UIMarkdownRendererInternal
    {
        ImFont *get_font() const override { return nullptr; };

        void open_url() const override {}

        bool get_image( image_info &nfo ) const override { return false; }

        void html_div( const std::string &dclass, bool e ) override
        {
            if( dclass == "red" )
            {
                if( e )
                {
                    m_table_border = false;
                    ImGui::PushStyleColor( ImGuiCol_Text, IM_COL32( 255, 0, 0, 255 ) );
                }
                else
                {
                    ImGui::PopStyleColor();
                    m_table_border = true;
                }
            }
        }
    };

    class UIMarkdown : public UIComponent
    {
      public:
        UIMarkdown() = default;

        UIMarkdown( string_t const &aText );

        void SetText( string_t const &aText );
        void SetTextColor( math::vec4 aColor );

        ImVec2 RequiredSize();

      protected:
        string_t           mText;
        ImVec4             mTextColor;
        UIMarkdownRenderer mRenderer;

      protected:
        void PushStyles();
        void PopStyles();

        void DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core