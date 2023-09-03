/*
 * UIMarkdownRendererInternal: Markdown for Dear ImGui using MD4C
 * (http://https://github.com/mekhontsev/UIMarkdownRendererInternal)
 *
 * Copyright (c) 2021 Dmitry Mekhontsev
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "MarkdownRenderer.h"
#include "Engine/Engine.h"
namespace SE::Core
{

    UIMarkdownRendererInternal::UIMarkdownRendererInternal()
    {
        m_md.abi_version = 0;

        m_md.flags = MD_FLAG_COLLAPSEWHITESPACE | MD_FLAG_TABLES | MD_FLAG_UNDERLINE | MD_FLAG_STRIKETHROUGH | MD_FLAG_TASKLISTS |
                     MD_FLAG_LATEXMATHSPANS | MD_FLAG_WIKILINKS;

        // clang-format off
        m_md.enter_block = []( MD_BLOCKTYPE t, void *d, void *u ) 
        { 
            return ( (UIMarkdownRendererInternal *)u )->block( t, d, true ); 
        };

        m_md.leave_block = []( MD_BLOCKTYPE t, void *d, void *u )
        { 
            return ( (UIMarkdownRendererInternal *)u )->block( t, d, false ); 
        };

        m_md.enter_span = []( MD_SPANTYPE t, void *d, void *u ) 
        { 
            return ( (UIMarkdownRendererInternal *)u )->span( t, d, true ); 
        };

        m_md.leave_span = []( MD_SPANTYPE t, void *d, void *u ) 
        { 
            return ( (UIMarkdownRendererInternal *)u )->span( t, d, false ); 
        };

        m_md.text = []( MD_TEXTTYPE t, const MD_CHAR *text, MD_SIZE size, void *u )
        { 
            return ( (UIMarkdownRendererInternal *)u )->text( t, text, text + size ); 
        };
        // clang-format on

        m_md.debug_log = nullptr;
        m_md.syntax    = nullptr;

        ////////////////////////////////////////////////////////////////////////////

        m_table_last_pos = ImVec2( 0, 0 );
    }

    // void UIMarkdownRendererInternal::LogBlockType( bool e, const char *str )
    // {
    //     if( e ) mBlockNestingLevel++;
    //     ImGui::PushStyleColor( ImGuiCol_Text, ImVec4( 1.0f, 1.0f, 0.0f, 1.0f ) );
    //     if( !e ) ImGui::NewLine();
    //     ImGui::Text( fmt::format( "{}{} {}", mBlockNestingLevel, e ? ">>>" : "<<<", str ).c_str() );
    //     if( !e )
    //     {
    //         mBlockNestingLevel--;
    //     }
    //     ImGui::PopStyleColor();
    // }

    // void UIMarkdownRendererInternal::BLOCK_UL( const MD_BLOCK_UL_DETAIL *d, bool e )
    // {

    //     if( e )
    //     {
    //         LogBlockType( e, "BLOCK_UL" );
    //         m_list_stack.push_back( list_info{ 0, d->mark, false } );
    //     }
    //     else
    //     {
    //         m_list_stack.pop_back();

    //         if( m_list_stack.empty() ) ImGui::NewLine();
    //         LogBlockType( e, "BLOCK_UL" );
    //     }
    // }

    // void UIMarkdownRendererInternal::BLOCK_OL( const MD_BLOCK_OL_DETAIL *d, bool e )
    // {

    //     if( e )
    //     {
    //         LogBlockType( e, "BLOCK_OL" );
    //         m_list_stack.push_back( list_info{ d->start, d->mark_delimiter, true } );
    //     }
    //     else
    //     {
    //         m_list_stack.pop_back();

    //         if( m_list_stack.empty() ) ImGui::NewLine();
    //         LogBlockType( e, "BLOCK_OL" );
    //     }
    // }

    // void UIMarkdownRendererInternal::BLOCK_LI( const MD_BLOCK_LI_DETAIL *, bool e )
    // {
    //     if( e )
    //     {
    //         LogBlockType( e, "BLOCK_LI" );
    //         ImGui::NewLine();

    //         list_info &nfo = m_list_stack.back();
    //         if( nfo.is_ol )
    //         {
    //             ImGui::Text( "%d%c", nfo.cur_ol++, nfo.delim );
    //             ImGui::SameLine();
    //         }
    //         else
    //         {
    //             if( nfo.delim == '*' )
    //             {
    //                 float cx = ImGui::GetCursorPosX();
    //                 cx -= ImGui::GetStyle().FramePadding.x * 2;
    //                 ImGui::SetCursorPosX( cx );
    //                 ImGui::Bullet();
    //             }
    //             else
    //             {
    //                 ImGui::Text( "%c", nfo.delim );
    //                 ImGui::SameLine();
    //             }
    //         }

    //         ImGui::Indent();
    //     }
    //     else
    //     {
    //         ImGui::Unindent();
    //         LogBlockType( e, "BLOCK_LI" );
    //     }
    // }

    void UIMarkdownRendererInternal::HRule::Render()
    {
        ImGui::NewLine();
        ImGui::Separator();

        UIMarkdownRendererInternal::Block::Render();
    }

    // void UIMarkdownRendererInternal::BLOCK_HR( bool e )
    // {
    //     if( e )
    //     {
    //         LogBlockType( e, "HR" );
    //     }
    //     else
    //     {
    //         ImGui::NewLine();
    //         ImGui::Separator();
    //         LogBlockType( e, "HR" );
    //     }
    // }

    void UIMarkdownRendererInternal::Heading::Render()
    {
        static FontFamilyFlags lLevelFonts[] = { FontFamilyFlags::H1, FontFamilyFlags::H2, FontFamilyFlags::H3 };

        ImGui::NewLine();
        if( mLevel <= 3 )
            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( lLevelFonts[mLevel - 1] );
        else
            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::NORMAL );

        UIMarkdownRendererInternal::Block::Render();
        SE::Core::Engine::GetInstance()->UIContext()->PopFont();
    }

    // void UIMarkdownRendererInternal::BLOCK_H( const MD_BLOCK_H_DETAIL *d, bool e )
    // {
    //     if( e )
    //     {
    //         LogBlockType( e, "H" );
    //         m_hlevel = d->level;
    //         ImGui::NewLine();
    //     }
    //     else
    //     {
    //         m_hlevel = 0;
    //     }

    //     set_font( e );

    //     if( !e )
    //     {
    //         if( d->level <= 2 )
    //         {
    //             ImGui::NewLine();
    //             ImGui::Separator();
    //         }
    //         LogBlockType( e, "H" );
    //     }
    // }

    // void UIMarkdownRendererInternal::BLOCK_DOC( bool e )
    // {
    //     //
    //     LogBlockType( e, "BLOCK_DOC" );
    // }

    void UIMarkdownRendererInternal::Quote::Render()
    {
        ImGui::NewLine();
        UIMarkdownRendererInternal::Block::Render();
    }

    // void UIMarkdownRendererInternal::BLOCK_QUOTE( bool e )
    // {
    //     //
    //     LogBlockType( e, "BLOCK_QUOTE" );
    // }

    void UIMarkdownRendererInternal::Code::Render()
    {
        ImGui::NewLine();
        UIMarkdownRendererInternal::Block::Render();
    }

    // void UIMarkdownRendererInternal::BLOCK_CODE( const MD_BLOCK_CODE_DETAIL *, bool e )
    // {
    //     m_is_code = e;

    //     LogBlockType( e, "BLOCK_CODE" );
    // }

    void UIMarkdownRendererInternal::Html::Render()
    {
        ImGui::NewLine();
        UIMarkdownRendererInternal::Block::Render();
    }

    // void UIMarkdownRendererInternal::BLOCK_HTML( bool e )
    // {
    //     //
    //     LogBlockType( e, "BLOCK_HTML" );
    // }

    void UIMarkdownRendererInternal::Paragraph::Render()
    {
        ImGui::NewLine();
        UIMarkdownRendererInternal::Block::Render();
    }

    void UIMarkdownRendererInternal::List::Render()
    {
        ImGui::NewLine();
        if( mIsOrderedList )
        {
            uint32_t lCurrentID = mStartIndex;
            ImGui::Indent();
            for( auto const &c : mChildren )
            {
                ImGui::Text( fmt::format( "{}", lCurrentID++ ).c_str() );
                ImGui::SameLine();
                c->Render();
                ImGui::NewLine();
            }
            ImGui::Unindent();
        }
        else
        {
            ImGui::Indent();
            for( auto const &c : mChildren )
            {
                ImGui::Bullet();
                ImGui::SameLine();
                c->Render();
                ImGui::NewLine();
            }
            ImGui::Unindent();
        }
    }

    void UIMarkdownRendererInternal::ListItem::Render()
    {
        UIMarkdownRendererInternal::Block::Render();
    }

    // void UIMarkdownRendererInternal::BLOCK_P( bool e )
    // {
    //     LogBlockType( e, "BLOCK_P" );

    //     if( !m_list_stack.empty() ) return;

    //     ImGui::NewLine();
    // }

    static ImVec2 GetContentAlignedposition( eHorizontalAlignment const &aHAlignment, ImVec2 aPosition, ImVec2 aContentSize,
                                             ImVec2 aSize )
    {
        ImVec2 lContentPosition{};
        switch( aHAlignment )
        {
        case eHorizontalAlignment::LEFT:
            lContentPosition.x = aPosition.x;
            break;
        case eHorizontalAlignment::RIGHT:
            lContentPosition.x = aPosition.x + ( aSize.x - aContentSize.x );
            break;
        case eHorizontalAlignment::CENTER:
        default:
            lContentPosition.x = aPosition.x + ( aSize.x - aContentSize.x ) * 0.5f;
            break;
        }

        lContentPosition.y = aPosition.y + ( aSize.y - aContentSize.y ) * 0.5f;

        return lContentPosition;
    }

    void UIMarkdownRendererInternal::Table::Render()
    {
        ImGui::NewLine();
        ImDrawList *dl = ImGui::GetWindowDrawList();

        int           x                 = 0;
        float         lCurrentX         = 0.0f;
        float         lCurrentY         = 0.0f;
        const ImColor c                 = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
        auto const   &lTablePosition    = ImGui::GetCursorPos();
        auto const   &lTableScrPosition = ImGui::GetCursorScreenPos();

        for( uint32_t i = 0; i < mColumns; i++ )
        {
            ImVec2 lCellSize{ mCells[i * ( mTableRows + 1 )].z, mCells[i * ( mTableRows + 1 )].w };
            auto   lTextPositionInCell =
                GetContentAlignedposition( eHorizontalAlignment::CENTER, ImVec2{ lCurrentX, lCurrentY },
                                           ImGui::CalcTextSize( mCellData[i * ( mTableRows + 1 )] ), lCellSize );
            ImGui::SetCursorPos( lTablePosition + lTextPositionInCell );
            ImGui::TextUnformatted( mCellData[i * ( mTableRows + 1 )] );
            lCurrentY += mCells[i * ( mTableRows + 1 )].w;

            for( uint32_t j = 1; j < mTableRows + 1; j++ )
            {
                // ImVec2 lContentSize{ mCells[i * ( mTableRows + 1 )].z, mCells[i * ( mTableRows + 1 ) + j].w };
                ImVec2 lCellSize{ mCells[i * ( mTableRows + 1 )].z, mCells[i * ( mTableRows + 1 ) + j].w };
                auto   lTextPositionInCell =
                    GetContentAlignedposition( mColumnAlignments[i], ImVec2{ lCurrentX, lCurrentY },
                                               ImGui::CalcTextSize( mCellData[i * ( mTableRows + 1 ) + j] ), lCellSize );
                ImGui::SetCursorPos( lTablePosition + lTextPositionInCell );
                ImGui::TextUnformatted( mCellData[i * ( mTableRows + 1 ) + j] );
                lCurrentY += mCells[i * ( mTableRows + 1 ) + j].w;
            }

            lCurrentX += mCells[i * ( mTableRows + 1 )].z;
            lCurrentY = 0.0f;
        }

        lCurrentX = lTableScrPosition.x;
        for( uint32_t i = 0; i <= mColumns; i++ )
        {
            dl->AddLine( ImVec2( lCurrentX, lTableScrPosition.y ), ImVec2( lCurrentX, lTableScrPosition.y + mHeight ), c, 1.0f );

            lCurrentX += ( i < mColumns ) ? mCells[i * ( mTableRows + 1 )].z : 0.0f;
        }

        lCurrentY = lTableScrPosition.y;
        for( uint32_t j = 0; j <= mTableRows + 1; j++ )
        {
            dl->AddLine( ImVec2( lTableScrPosition.x, lCurrentY ), ImVec2( lTableScrPosition.x + mWidth, lCurrentY ), c,
                         j == 1 ? 2.0f : 1.0f );

            lCurrentY += ( j <= mTableRows ) ? mCells[j].w : 0.0f;
        }
    }

    void UIMarkdownRendererInternal::Table::ComputeColumnSizes()
    {
        mCells    = vector_t<ImVec4>( ( mTableRows + 1 ) * mColumns );
        mCellData = vector_t<const char *>( ( mTableRows + 1 ) * mColumns );
        mHeight   = 0.0f;

        int x = 0;
        for( uint32_t i = 0; i < mColumns; i++ )
        {
            auto const S = ImGui::CalcTextSize( mHeader[i].c_str() ) + ImVec2{ 10.0f, 10.0f };

            mCells[x]    = ImVec4( 0.0f, 0.0f, S.x, S.y );
            mCellData[x] = mHeader[i].c_str();
            x++;

            for( uint32_t j = 0; j < mBody[i].size(); j++ )
            {
                auto const S = ImGui::CalcTextSize( mBody[i][j].c_str() ) + ImVec2{ 10.0f, 10.0f };

                mCells[x]    = ImVec4( 0.0f, 0.0f, S.x, S.y );
                mCellData[x] = mBody[i][j].c_str();
                x++;
            }
        }

        x = 0;
        for( uint32_t i = 0; i < mColumns; i++ )
        {
            int   lColStart = x;
            float lWidth    = mCells[x++].z;
            for( uint32_t j = 1; j < mTableRows + 1; j++ )
            {
                lWidth = std::max( lWidth, mCells[x++].z );
            }

            mWidth += lWidth;
            for( uint32_t j = 0; j < mTableRows + 1; j++ )
            {
                mCells[j + lColStart].z = lWidth;
            }
        }

        for( uint32_t j = 0; j < mTableRows + 1; j++ )
        {
            float lHeight = mCells[j].w;
            for( uint32_t i = 0; i < mColumns; i++ )
            {
                lHeight = std::max( lHeight, mCells[i * ( mTableRows + 1 ) + j].w );
            }

            mHeight += lHeight;
            for( uint32_t i = 0; i < mColumns; i++ )
            {
                mCells[i * ( mTableRows + 1 ) + j].w = lHeight;
            }
        }
    }

    // void UIMarkdownRendererInternal::BLOCK_TABLE( const MD_BLOCK_TABLE_DETAIL *, bool e )
    // {
    //     if( e )
    //     {
    //         LogBlockType( e, "BLOCK_TABLE" );
    //         m_table_row_pos.clear();
    //         m_table_col_pos.clear();

    //         m_table_last_pos = ImGui::GetCursorPos();
    //     }
    //     else
    //     {

    //         ImGui::NewLine();

    //         if( m_table_border )
    //         {

    //             m_table_last_pos.y = ImGui::GetCursorPos().y;

    //             m_table_col_pos.push_back( m_table_last_pos.x );
    //             m_table_row_pos.push_back( m_table_last_pos.y );

    //             const ImVec2 wp = ImGui::GetWindowPos();
    //             const ImVec2 sp = ImGui::GetStyle().ItemSpacing;
    //             const float  wx = wp.x + sp.x / 2;
    //             const float  wy = wp.y - sp.y / 2 - ImGui::GetScrollY();

    //             for( int i = 0; i < m_table_col_pos.size(); ++i )
    //             {
    //                 m_table_col_pos[i] += wx;
    //             }

    //             for( int i = 0; i < m_table_row_pos.size(); ++i )
    //             {
    //                 m_table_row_pos[i] += wy;
    //             }

    //             ////////////////////////////////////////////////////////////////////

    //             const ImColor c = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];

    //             ImDrawList *dl = ImGui::GetWindowDrawList();

    //             const float xmin = m_table_col_pos.front();
    //             const float xmax = m_table_col_pos.back();
    //             for( int i = 0; i < m_table_row_pos.size(); ++i )
    //             {
    //                 const float p = m_table_row_pos[i];
    //                 dl->AddLine( ImVec2( xmin, p ), ImVec2( xmax, p ), c, i == 1 && m_table_header_highlight ? 2.0f : 1.0f );
    //             }

    //             const float ymin = m_table_row_pos.front();
    //             const float ymax = m_table_row_pos.back();
    //             for( int i = 0; i < m_table_col_pos.size(); ++i )
    //             {
    //                 const float p = m_table_col_pos[i];
    //                 dl->AddLine( ImVec2( p, ymin ), ImVec2( p, ymax ), c, 1.0f );
    //             }
    //         }
    //         LogBlockType( e, "BLOCK_TABLE" );
    //     }
    // }

    // void UIMarkdownRendererInternal::BLOCK_THEAD( bool e )
    // {
    //     LogBlockType( e, "BLOCK_THEAD" );

    //     m_is_table_header = e;

    //     if( m_table_header_highlight ) set_font( e );
    // }

    // void UIMarkdownRendererInternal::BLOCK_TBODY( bool e )
    // {
    //     LogBlockType( e, "BLOCK_TBODY" );

    //     m_is_table_body = e;
    // }

    // void UIMarkdownRendererInternal::BLOCK_TR( bool e )
    // {
    //     ImGui::SetCursorPosY( m_table_last_pos.y );

    //     LogBlockType( e, "BLOCK_TR" );
    //     if( e )
    //     {
    //         m_table_next_column = 0;
    //         ImGui::NewLine();
    //         m_table_row_pos.push_back( ImGui::GetCursorPosY() );
    //     }
    // }

    // void UIMarkdownRendererInternal::BLOCK_TH( const MD_BLOCK_TD_DETAIL *d, bool e )
    // {
    //     LogBlockType( e, "BLOCK_TH" );

    //     BLOCK_TD( d, e );
    // }

    // void UIMarkdownRendererInternal::BLOCK_TD( const MD_BLOCK_TD_DETAIL *, bool e )
    // {

    //     if( e )
    //     {
    //         LogBlockType( e, "BLOCK_TD" );
    //         if( m_table_next_column < m_table_col_pos.size() )
    //         {
    //             ImGui::SetCursorPosX( m_table_col_pos[m_table_next_column] );
    //         }
    //         else
    //         {
    //             m_table_col_pos.push_back( m_table_last_pos.x );
    //         }

    //         ++m_table_next_column;

    //         ImGui::Indent( m_table_col_pos[m_table_next_column - 1] );
    //         ImGui::SetCursorPos( ImVec2( m_table_col_pos[m_table_next_column - 1], m_table_row_pos.back() ) );
    //     }
    //     else
    //     {
    //         const ImVec2 p = ImGui::GetCursorPos();
    //         ImGui::Unindent( m_table_col_pos[m_table_next_column - 1] );
    //         ImGui::SetCursorPosX( p.x );
    //         if( p.y > m_table_last_pos.y ) m_table_last_pos.y = p.y;
    //         LogBlockType( e, "BLOCK_TD" );
    //     }

    //     ImGui::TextUnformatted( "" );

    //     if( !m_table_border && e && m_table_next_column == 1 )
    //     {
    //         ImGui::SameLine( 0.0f, 0.0f );
    //     }
    //     else
    //     {
    //         ImGui::SameLine();
    //     }
    // }

    ////////////////////////////////////////////////////////////////////////////////
    void UIMarkdownRendererInternal::set_href( bool e, const MD_ATTRIBUTE &src )
    {
        if( e )
        {
            m_href.assign( src.text, src.size );
        }
        else
        {
            m_href.clear();
        }
    }

    void UIMarkdownRendererInternal::set_font( bool e )
    {

        if( m_is_table_header )
        {
            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::BOLD );
            return;
        }

        if( !e )
        {
            SE::Core::Engine::GetInstance()->UIContext()->PopFont();
            return;
        }

        switch( m_hlevel )
        {
        case 0:
            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( m_is_strong ? FontFamilyFlags::BOLD
                                                                                      : FontFamilyFlags::NORMAL );
            break;
        case 1:
            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::H1 );
            break;
        case 2:
            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::H2 );
            break;
        case 3:
            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::H3 );
            break;
        default:
            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::NORMAL );
            break;
        }
    }

    void UIMarkdownRendererInternal::set_color( bool e )
    {
        if( e )
        {
            ImGui::PushStyleColor( ImGuiCol_Text, get_color() );
        }
        else
        {
            ImGui::PopStyleColor();
        }
    }

    void UIMarkdownRendererInternal::line( ImColor c, bool under )
    {
        ImVec2 mi = ImGui::GetItemRectMin();
        ImVec2 ma = ImGui::GetItemRectMax();

        if( !under )
        {
            ma.y -= ImGui::GetFontSize() / 2;
        }

        mi.y = ma.y;

        ImGui::GetWindowDrawList()->AddLine( mi, ma, c, 1.0f );
    }

    void UIMarkdownRendererInternal::Text::Render()
    {
        const float scale   = ImGui::GetIO().FontGlobalScale;
        const char *str     = mStart;
        const char *str_end = mEnd;
        bool        is_lf   = false;

        while( str < str_end )
        {
            const char *te = str_end;

            te = ImGui::GetFont()->CalcWordWrapPositionA( scale, str, str_end, ImGui::GetContentRegionAvail().x );

            if( te == str )
                ++te;
            if( te > str && *( te - 1 ) == '\n' )
                is_lf = true;

            ImGui::TextUnformatted( str, te );

            str = te;

            while( str < str_end && *str == ' ' )
                ++str;
        }

        if( !is_lf )
            ImGui::SameLine( 0.0f, 0.0f );
    }

    // void UIMarkdownRendererInternal::SPAN_A( const MD_SPAN_A_DETAIL *d, bool e )
    // {
    //     LogBlockType( e, "SPAN_A" );
    //     set_href( e, d->href );
    //     set_color( e );
    // }

    void UIMarkdownRendererInternal::Emphasis::Render()
    {
        SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::EM );
        UIMarkdownRendererInternal::Block::Render();
        SE::Core::Engine::GetInstance()->UIContext()->PopFont();
    }

    // void UIMarkdownRendererInternal::SPAN_EM( bool e )
    // {
    //     LogBlockType( e, "SPAN_EM" );
    //     m_is_em = e;
    //     set_font( e );
    // }

    void UIMarkdownRendererInternal::Strong::Render()
    {
        SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::BOLD );
        UIMarkdownRendererInternal::Block::Render();
        SE::Core::Engine::GetInstance()->UIContext()->PopFont();
    }

    // void UIMarkdownRendererInternal::SPAN_STRONG( bool e )
    // {
    //     LogBlockType( e, "SPAN_STRONG" );
    //     m_is_strong = e;
    //     set_font( e );
    // }

    // void UIMarkdownRendererInternal::SPAN_IMG( const MD_SPAN_IMG_DETAIL *d, bool e )
    // {
    //     LogBlockType( e, "SPAN_IMG" );
    //     m_is_image = e;

    //     set_href( e, d->src );

    //     if( e )
    //     {
    //         image_info nfo;
    //         if( get_image( nfo ) )
    //         {
    //             const float scale = ImGui::GetIO().FontGlobalScale;
    //             nfo.size.x *= scale;
    //             nfo.size.y *= scale;

    //             ImVec2 const csz = ImGui::GetContentRegionAvail();
    //             if( nfo.size.x > csz.x )
    //             {
    //                 const float r = nfo.size.y / nfo.size.x;
    //                 nfo.size.x    = csz.x;
    //                 nfo.size.y    = csz.x * r;
    //             }

    //             ImGui::Image( nfo.texture_id, nfo.size, nfo.uv0, nfo.uv1, nfo.col_tint, nfo.col_border );

    //             if( ImGui::IsItemHovered() )
    //             {

    //                 // if (d->title.size) {
    //                 //	ImGui::SetTooltip("%.*s", (int)d->title.size, d->title.text);
    //                 // }

    //                 if( ImGui::IsMouseReleased( 0 ) )
    //                 {
    //                     open_url();
    //                 }
    //             }
    //         }
    //     }
    // }

    // void UIMarkdownRendererInternal::SPAN_CODE( bool e )
    // {
    //     //
    //     LogBlockType( e, "SPAN_CODE" );
    // }

    // void UIMarkdownRendererInternal::SPAN_LATEXMATH( bool e )
    // {
    //     //
    //     LogBlockType( e, "SPAN_LATEXMATH" );
    // }

    // void UIMarkdownRendererInternal::SPAN_LATEXMATH_DISPLAY( bool e )
    // {
    //     //
    //     LogBlockType( e, "SPAN_LATEXMATH_DISPLAY" );
    // }

    // void UIMarkdownRendererInternal::SPAN_WIKILINK( const MD_SPAN_WIKILINK_DETAIL *, bool e )
    // {
    //     //
    //     LogBlockType( e, "SPAN_WIKILINK" );
    // }

    // void UIMarkdownRendererInternal::SPAN_U( bool e )
    // {
    //     //
    //     LogBlockType( e, "SPAN_U" );
    //     m_is_underline = e;
    // }

    // void UIMarkdownRendererInternal::SPAN_DEL( bool e )
    // {
    //     //
    //     LogBlockType( e, "SPAN_DEL" );
    //     m_is_strikethrough = e;
    // }

    void UIMarkdownRendererInternal::render_text( const char *str, const char *str_end )
    {
        const float       scale = ImGui::GetIO().FontGlobalScale;
        const ImGuiStyle &s     = ImGui::GetStyle();
        bool              is_lf = false;

        while( !m_is_image && str < str_end )
        {
            const char *te = str_end;

            if( !m_is_table_header )
            {
                float wl = ImGui::GetContentRegionAvail().x - ( mLeftMargin + mRightMargin );

                if( m_is_table_body )
                {
                    wl = ( m_table_next_column < m_table_col_pos.size() ? m_table_col_pos[m_table_next_column] : m_table_last_pos.x );
                    wl -= ImGui::GetCursorPosX();
                }

                te = ImGui::GetFont()->CalcWordWrapPositionA( scale, str, str_end, wl );

                if( te == str )
                    ++te;
            }

            ImGui::TextUnformatted( str, te );

            if( te > str && *( te - 1 ) == '\n' )
            {
                is_lf = true;
            }

            if( !m_href.empty() )
            {

                ImVec4 c;
                if( ImGui::IsItemHovered() )
                {
                    ImGui::SetTooltip( "%s", m_href.c_str() );

                    c = s.Colors[ImGuiCol_ButtonHovered];
                    if( ImGui::IsMouseReleased( 0 ) )
                    {
                        open_url();
                    }
                }
                else
                {
                    c = s.Colors[ImGuiCol_Button];
                }

                line( c, true );
            }

            if( m_is_underline )
            {
                line( s.Colors[ImGuiCol_Text], true );
            }

            if( m_is_strikethrough )
            {
                line( s.Colors[ImGuiCol_Text], false );
            }

            str = te;

            while( str < str_end && *str == ' ' )
                ++str;
        }

        if( !is_lf )
            ImGui::SameLine( 0.0f, 0.0f );
    }

    bool UIMarkdownRendererInternal::render_entity( const char *str, const char *str_end )
    {
        const size_t sz = str_end - str;
        if( strncmp( str, "&nbsp;", sz ) == 0 )
        {
            ImGui::TextUnformatted( "" );
            ImGui::SameLine();
            return true;
        }
        return false;
    }

    static bool skip_spaces( const string_t &d, size_t &p )
    {
        for( ; p < d.length(); ++p )
        {
            if( d[p] != ' ' && d[p] != '\t' )
            {
                break;
            }
        }
        return p < d.length();
    }

    static string_t get_div_class( const char *str, const char *str_end )
    {
        if( str_end <= str )
            return "";

        string_t d( str, str_end - str );
        if( d.back() == '>' )
            d.pop_back();

        const char attr[] = "class";
        size_t     p      = d.find( attr );
        if( p == string_t::npos )
            return "";
        p += sizeof( attr ) - 1;

        if( !skip_spaces( d, p ) )
            return "";

        if( d[p] != '=' )
            return "";
        ++p;

        if( !skip_spaces( d, p ) )
            return "";

        bool has_q = false;

        if( d[p] == '"' || d[p] == '\'' )
        {
            has_q = true;
            ++p;
        }
        if( p == d.length() )
            return "";

        if( !has_q )
        {
            if( !skip_spaces( d, p ) )
                return "";
        }

        size_t pe;
        for( pe = p; pe < d.length(); ++pe )
        {

            const char c = d[pe];

            if( has_q )
            {
                if( c == '"' || c == '\'' )
                {
                    break;
                }
            }
            else
            {
                if( c == ' ' || c == '\t' )
                {
                    break;
                }
            }
        }

        return d.substr( p, pe - p );
    }

    bool UIMarkdownRendererInternal::check_html( const char *str, const char *str_end )
    {
        const size_t sz = str_end - str;

        if( strncmp( str, "<br>", sz ) == 0 )
        {
            ImGui::NewLine();
            return true;
        }
        if( strncmp( str, "<hr>", sz ) == 0 )
        {
            ImGui::Separator();
            return true;
        }
        if( strncmp( str, "<u>", sz ) == 0 )
        {
            m_is_underline = true;
            return true;
        }
        if( strncmp( str, "</u>", sz ) == 0 )
        {
            m_is_underline = false;
            return true;
        }

        const size_t div_sz = 4;
        if( strncmp( str, "<div", sz > div_sz ? div_sz : sz ) == 0 )
        {
            m_div_stack.emplace_back( get_div_class( str + div_sz, str_end ) );
            html_div( m_div_stack.back(), true );
            return true;
        }
        if( strncmp( str, "</div>", sz ) == 0 )
        {
            if( m_div_stack.empty() )
                return false;
            html_div( m_div_stack.back(), false );
            m_div_stack.pop_back();
            return true;
        }
        return false;
    }

    void UIMarkdownRendererInternal::html_div( const string_t &dclass, bool e )
    {
    }

    int UIMarkdownRendererInternal::text( MD_TEXTTYPE type, const char *str, const char *str_end )
    {
        if( mCurrentTable != nullptr )
        {
            if( mCurrentTable->mFillHeader )
            {
                mCurrentTable->mHeader[mCurrentTable->mCurrentColumn - 1] = string_t( str, str_end - str );
            }

            if( mCurrentTable->mFillBody )
            {
                mCurrentTable->mBody[mCurrentTable->mCurrentColumn - 1][mCurrentTable->mCurrentRow - 1] =
                    string_t( str, str_end - str );
            }

            return 0;
        }

        AppendBlock<Text>( type, str, str_end );

        switch( type )
        {
        case MD_TEXT_NORMAL:
        {
            render_text( str, str_end );
            break;
        }
        case MD_TEXT_CODE:
        {
            render_text( str, str_end );
            break;
        }
        case MD_TEXT_NULLCHAR:
        {
            break;
        }
        case MD_TEXT_BR:
        {
            ImGui::NewLine();
            break;
        }
        case MD_TEXT_SOFTBR:
        {
            soft_break();
            break;
        }
        case MD_TEXT_ENTITY:
        {
            if( !render_entity( str, str_end ) )
            {
                render_text( str, str_end );
            };
            break;
        }
        case MD_TEXT_HTML:
        {
            if( !check_html( str, str_end ) )
            {
                render_text( str, str_end );
            }
            break;
        }
        case MD_TEXT_LATEXMATH:
        {
            render_text( str, str_end );
            break;
        }
        default:
            break;
        }

        if( m_is_table_header )
        {
            const float x = ImGui::GetCursorPosX();
            if( x > m_table_last_pos.x )
                m_table_last_pos.x = x;
        }

        return 0;
    }

    int UIMarkdownRendererInternal::block( MD_BLOCKTYPE type, void *d, bool e )
    {
        if( !e )
        {
            mCurrentBlock = mCurrentBlock->mParent;

            if( type == MD_BLOCK_TABLE )
            {
                mCurrentTable->ComputeColumnSizes();
                mCurrentTable = nullptr;
            }

            if( ( type == MD_BLOCK_OL ) || ( type == MD_BLOCK_UL ) )
                mListStack.pop_back();

            return 0;
        }

        switch( type )
        {
        case MD_BLOCK_DOC:
            PushBlock<Document>();
            break;
        case MD_BLOCK_QUOTE:
            PushBlock<Quote>();
            break;
        case MD_BLOCK_UL:
            PushBlock<List>( (MD_BLOCK_UL_DETAIL *)d );
            mListStack.push_back( std::reinterpret_pointer_cast<List>( mCurrentBlock ) );
            break;
        case MD_BLOCK_OL:
            PushBlock<List>( (MD_BLOCK_OL_DETAIL *)d );
            mListStack.push_back( std::reinterpret_pointer_cast<List>( mCurrentBlock ) );
            break;
        case MD_BLOCK_LI:
            PushBlock<ListItem>( (MD_BLOCK_LI_DETAIL *)d );
            break;
        case MD_BLOCK_HR:
            PushBlock<HRule>();
            break;
        case MD_BLOCK_H:
            PushBlock<Heading>( (MD_BLOCK_H_DETAIL *)d );
            break;
        case MD_BLOCK_CODE:
            PushBlock<Code>( (MD_BLOCK_CODE_DETAIL *)d );
            break;
        case MD_BLOCK_HTML:
            PushBlock<Html>();
            break;
        case MD_BLOCK_P:
            PushBlock<Paragraph>();
            break;
        case MD_BLOCK_TABLE:
            PushBlock<Table>( (MD_BLOCK_TABLE_DETAIL *)d );
            mCurrentTable = std::reinterpret_pointer_cast<Table>( mCurrentBlock );
            for( uint32_t j = 0; j < mCurrentTable->mColumns; j++ )
            {
                mCurrentTable->mBody.push_back( vector_t<string_t>() );

                for( uint32_t i = 0; i < mCurrentTable->mTableRows; i++ )
                {
                    mCurrentTable->mBody.back().push_back( string_t( "" ) );
                }
            }

            for( uint32_t i = 0; i < mCurrentTable->mColumns; i++ )
            {
                mCurrentTable->mHeader.push_back( "" );
            }

            mCurrentTable->mCurrentRow    = 0;
            mCurrentTable->mCurrentColumn = 0;
            break;
        case MD_BLOCK_THEAD:
            PushBlock<TableHeader>();
            std::reinterpret_pointer_cast<Table>( mCurrentTable )->mFillHeader = true;
            std::reinterpret_pointer_cast<Table>( mCurrentTable )->mFillBody   = false;
            break;
        case MD_BLOCK_TBODY:
            PushBlock<TableBody>();
            std::reinterpret_pointer_cast<Table>( mCurrentTable )->mFillHeader = false;
            std::reinterpret_pointer_cast<Table>( mCurrentTable )->mFillBody   = true;
            break;
        case MD_BLOCK_TR:
            PushBlock<TableRow>();

            if( std::reinterpret_pointer_cast<Table>( mCurrentTable )->mFillBody )
            {
                mCurrentTable->mCurrentRow++;
                mCurrentTable->mCurrentColumn = 0;
            }
            break;
        case MD_BLOCK_TH:
            PushBlock<TableData>( (MD_BLOCK_TD_DETAIL *)d );
            mCurrentTable->mCurrentColumn++;
            switch( ( (MD_BLOCK_TD_DETAIL *)d )->align )
            {
            case MD_ALIGN::MD_ALIGN_LEFT:
                mCurrentTable->mColumnAlignments.push_back( eHorizontalAlignment::LEFT );
                break;
            case MD_ALIGN::MD_ALIGN_RIGHT:
                mCurrentTable->mColumnAlignments.push_back( eHorizontalAlignment::RIGHT );
                break;
            case MD_ALIGN::MD_ALIGN_CENTER:
            case MD_ALIGN::MD_ALIGN_DEFAULT:
            default:
                mCurrentTable->mColumnAlignments.push_back( eHorizontalAlignment::CENTER );
                break;
            }
            break;
        case MD_BLOCK_TD:
            PushBlock<TableData>( (MD_BLOCK_TD_DETAIL *)d );
            mCurrentTable->mCurrentColumn++;
            break;
        default:
            assert( false );
            break;
        }

        return 0;
    } // namespace SE::Core

    int UIMarkdownRendererInternal::span( MD_SPANTYPE type, void *d, bool e )
    {
        if( !e )
        {
            mCurrentBlock = mCurrentBlock->mParent;
            return 0;
        }

        switch( type )
        {
        case MD_SPAN_EM:
            PushBlock<Emphasis>();
            // SPAN_EM( e );
            break;
        case MD_SPAN_STRONG:
            PushBlock<Strong>();
            // SPAN_STRONG( e );
            break;
        case MD_SPAN_A:
            PushBlock<Link>( (MD_SPAN_A_DETAIL *)d );
            // SPAN_A( (MD_SPAN_A_DETAIL *)d, e );
            break;
        case MD_SPAN_IMG:
            PushBlock<Image>( (MD_SPAN_IMG_DETAIL *)d );
            // SPAN_IMG( (MD_SPAN_IMG_DETAIL *)d, e );
            break;
        case MD_SPAN_CODE:
            PushBlock<InlineCode>();
            // SPAN_CODE( e );
            break;
        case MD_SPAN_DEL:
            PushBlock<StrikeThrough>();
            // SPAN_DEL( e );
            break;
        case MD_SPAN_LATEXMATH:
            PushBlock<LaTeXMath>();
            // SPAN_LATEXMATH( e );
            break;
        case MD_SPAN_LATEXMATH_DISPLAY:
            PushBlock<LaTeXMath>();
            // SPAN_LATEXMATH_DISPLAY( e );
            break;
        case MD_SPAN_WIKILINK:
            PushBlock<WikiLink>( (MD_SPAN_WIKILINK_DETAIL *)d );
            // SPAN_WIKILINK( (MD_SPAN_WIKILINK_DETAIL *)d, e );
            break;
        case MD_SPAN_U:
            PushBlock<Underline>();
            // SPAN_U( e );
            break;
        default:
            assert( false );
            break;
        }

        return 0;
    }

    int UIMarkdownRendererInternal::print( const char *str, const char *str_end )
    {
        if( str >= str_end )
            return 0;

        if( mRootBlock )
        {
            SE::Core::Engine::GetInstance()->UIContext()->PushFontFamily( FontFamilyFlags::NORMAL );
            mRootBlock->Render();
            SE::Core::Engine::GetInstance()->UIContext()->PopFont();

            return 0;
        }

        mRootBlock    = New<Block>();
        mCurrentBlock = mRootBlock;
        md_parse( str, (MD_SIZE)( str_end - str ), &m_md, this );

        return 0;
    }

    ////////////////////////////////////////////////////////////////////////////////

    ImFont *UIMarkdownRendererInternal::get_font() const
    {
        return nullptr; // default font
    };

    ImVec4 UIMarkdownRendererInternal::get_color() const
    {
        if( !m_href.empty() )
        {
            return ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered];
        }
        return ImGui::GetStyle().Colors[ImGuiCol_Text];
    }

    bool UIMarkdownRendererInternal::get_image( image_info &nfo ) const
    {
        // Use m_href to identify images

        // Example - Imgui font texture
        nfo.texture_id = ImGui::GetIO().Fonts->TexID;
        nfo.size       = { 100, 50 };
        nfo.uv0        = { 0, 0 };
        nfo.uv1        = { 1, 1 };
        nfo.col_tint   = { 1, 1, 1, 1 };
        nfo.col_border = { 0, 0, 0, 0 };

        return true;
    };

    void UIMarkdownRendererInternal::open_url() const
    {
        // Example:
    }

    void UIMarkdownRendererInternal::soft_break()
    {
        // Example:
    }

} // namespace SE::Core
