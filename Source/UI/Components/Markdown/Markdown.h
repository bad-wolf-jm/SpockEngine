#pragma once

#include "UI/Components/Component.h"
#include "MarkdownRenderer.h"

namespace SE::Core
{

    // // Fonts and images (ImTextureID) must be loaded in other place
    // // see https://github.com/ocornut/imgui/blob/master/docs/FONTS.md
    // extern ImFont     *g_font_regular;
    // extern ImFont     *g_font_bold;
    // extern ImFont     *g_font_bold_large;
    // extern ImTextureID g_texture1;

    struct UIMarkdownRenderer : public UIMarkdownRendererInternal
    {
        ImFont *get_font() const override
        {
            return nullptr;
            // if( m_is_table_header )
            // {
            //     return g_font_bold;
            // }
            // switch( m_hlevel )
            // {
            // case 0: return m_is_strong ? g_font_bold : g_font_regular;
            // case 1: return g_font_bold_large;
            // default: return g_font_bold;
            // }
        };

        void open_url() const override {}

        bool get_image( image_info &nfo ) const override
        {
            // use m_href to identify images
            // nfo.texture_id = g_texture1;
            // nfo.size       = { 40, 20 };
            // nfo.uv0        = { 0, 0 };
            // nfo.uv1        = { 1, 1 };
            // nfo.col_tint   = { 1, 1, 1, 1 };
            // nfo.col_border = { 0, 0, 0, 0 };
            return false;
        }

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

    // // call this function to render your markdown
    // void markdown( const char *str, const char *str_end )
    // {
    //     static UIMarkdownRenderer s_printer;
    //     s_printer.print( str, str_end );
    // }

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