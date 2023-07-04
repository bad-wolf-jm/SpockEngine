#pragma once
/*
 * imgui_md: Markdown for Dear ImGui using MD4C
 * (http://https://github.com/mekhontsev/imgui_md)
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

#include "UI/Components/Component.h"
#include "imgui.h"
#include "md4c.h"
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace SE::Core
{
    class UIMarkdownRendererInternal
    {
      public:
        UIMarkdownRendererInternal();
        virtual ~UIMarkdownRendererInternal(){};

        // returns 0 on success
        int print( const char *str, const char *str_end );

        // for example, these flags can be changed in div callback

        // draw border
        bool m_table_border = true;
        // render header in a different way than other rows
        bool m_table_header_highlight = true;

      protected:
        void BLOCK_DOC( bool );
        void BLOCK_QUOTE( bool );
        void BLOCK_UL( const MD_BLOCK_UL_DETAIL *, bool );
        void BLOCK_OL( const MD_BLOCK_OL_DETAIL *, bool );
        void BLOCK_LI( const MD_BLOCK_LI_DETAIL *, bool );
        void BLOCK_HR( bool e );
        void BLOCK_H( const MD_BLOCK_H_DETAIL *d, bool e );
        void BLOCK_CODE( const MD_BLOCK_CODE_DETAIL *, bool );
        void BLOCK_HTML( bool );
        void BLOCK_P( bool );
        void BLOCK_TABLE( const MD_BLOCK_TABLE_DETAIL *, bool );
        void BLOCK_THEAD( bool );
        void BLOCK_TBODY( bool );
        void BLOCK_TR( bool );
        void BLOCK_TH( const MD_BLOCK_TD_DETAIL *, bool );
        void BLOCK_TD( const MD_BLOCK_TD_DETAIL *, bool );

        void SPAN_EM( bool e );
        void SPAN_STRONG( bool e );
        void SPAN_A( const MD_SPAN_A_DETAIL *d, bool e );
        void SPAN_IMG( const MD_SPAN_IMG_DETAIL *, bool );
        void SPAN_CODE( bool );
        void SPAN_DEL( bool );
        void SPAN_LATEXMATH( bool );
        void SPAN_LATEXMATH_DISPLAY( bool );
        void SPAN_WIKILINK( const MD_SPAN_WIKILINK_DETAIL *, bool );
        void SPAN_U( bool );

        ////////////////////////////////////////////////////////////////////////////

        enum eBlockType
        {
            DOCUMENT,
            QUOTE,
            UORDERED_LIST,
            ORDERED_LIST,
            LIST_ITEM,
            HRULE,
            HEADING,
            CODE,
            HTML,
            PARAGRAPH,
            TABLE,
            TABLE_HEADER,
            TABLE_BODY,
            TABLE_ROW,
            TABLE_DATA
        };

        struct Block
        {
            eBlockType              mType;
            ImVec2                  mPosition;
            ImVec2                  mSize;
            Ref<Block>              mParent = nullptr;
            std::vector<Ref<Block>> mChildren;
            uint32_t                 mDepth = 0;

            Block() = default;

            Block( Ref<Block> aParent, eBlockType aType )
                : mType{ aType }
                , mParent{ aParent }
            {
            }

            virtual void Render()
            {
                for( auto const &c : mChildren ) c->Render();
            }
        };

        struct Document : public Block
        {
            Document( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }
        };

        struct Quote : public Block
        {
            Quote( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render();
        };

        struct UnorderedList : public Block
        {
            char mMarker;

            UnorderedList( Ref<Block> aParent, eBlockType aType, MD_BLOCK_UL_DETAIL *d )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct OrderedList : public Block
        {
            uint32_t mStartIndex;
            OrderedList( Ref<Block> aParent, eBlockType aType, MD_BLOCK_OL_DETAIL *d )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct ListItem : public Block
        {
            bool     mIsTask;
            bool     mIsChecked;
            uint32_t mTTaskMarkOffset;

            ListItem( Ref<Block> aParent, eBlockType aType, MD_BLOCK_LI_DETAIL *d )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct HRule : public Block
        {
            HRule( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render();
        };

        struct Heading : public Block
        {
            uint32_t mLevel;

            Heading( Ref<Block> aParent, eBlockType aType, MD_BLOCK_H_DETAIL *d )
                : Block( aParent, aType )
                , mLevel{ d->level }
            {
            }

            void Render();
        };

        struct Code : public Block
        {
            char *mLanguage;
            char  mFenceChar;

            Code( Ref<Block> aParent, eBlockType aType, MD_BLOCK_CODE_DETAIL *d )
                : Block( aParent, aType )
            {
            }

            void Render();
        };

        struct Html : public Block
        {
            Html( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render();
        };

        struct Paragraph : public Block
        {
            Paragraph( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render();
        };

        struct Table : public Block
        {
            uint32_t mColumns;
            uint32_t mTableRows;

            bool mFillHeader = false;
            bool mFillBody = false;

            std::vector<string_t> mHeader;
            std::vector<std::vector<string_t>> mBody;

            std::vector<ImVec4> mCells;
            std::vector<const char*> mCellData;

            int32_t mCurrentRow = -1;
            int32_t mCurrentColumn = -1;

            Table( Ref<Block> aParent, eBlockType aType, MD_BLOCK_TABLE_DETAIL *d )
                : Block( aParent, aType )
                , mColumns{d->col_count}
                , mTableRows{d->body_row_count}
            {
            }

            void Render();
            void ComputeColumnSizes();
        };

        struct TableHeader : public Block
        {
            TableHeader( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct TableBody : public Block
        {
            TableBody( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct TableRow : public Block
        {
            TableRow( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct TableData : public Block
        {
            eHorizontalAlignment mAlign;

            TableData( Ref<Block> aParent, eBlockType aType, MD_BLOCK_TD_DETAIL *d )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct Text : public Block
        {
            MD_TEXTTYPE mTextType;

            const char *mStart;
            const char *mEnd;

            Text( Ref<Block> aParent, eBlockType aType, MD_TEXTTYPE d, const char *aStrBegin, const char *aStrEnd )
                : Block( aParent, aType )
                , mStart{ aStrBegin }
                , mEnd{ aStrEnd }
            {
            }

            void Render();
        };
        struct Emphasis : public Block
        {
            Emphasis( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render();
        };

        struct Strong : public Block
        {
            Strong( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render();
        };

        struct Underline : public Block
        {
            Underline( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct Link : public Block
        {
            Link( Ref<Block> aParent, eBlockType aType, MD_SPAN_A_DETAIL *d )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct Image : public Block
        {
            Image( Ref<Block> aParent, eBlockType aType, MD_SPAN_IMG_DETAIL *d )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct InlineCode : public Block
        {
            InlineCode( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct StrikeThrough : public Block
        {
            StrikeThrough( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct LaTeXMath : public Block
        {
            LaTeXMath( Ref<Block> aParent, eBlockType aType )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        struct WikiLink : public Block
        {
            WikiLink( Ref<Block> aParent, eBlockType aType, MD_SPAN_WIKILINK_DETAIL *d )
                : Block( aParent, aType )
            {
            }

            void Render() {}
        };

        Ref<Block> mRootBlock;
        Ref<Block> mCurrentBlock;
        Ref<Table> mCurrentTable;

        template <typename _Ty, typename... _Args>
        void PushBlock( _Args... aArgs )
        {
            uint32_t d = mCurrentBlock->mDepth;
            mCurrentBlock->mChildren.push_back( New<_Ty>( mCurrentBlock, eBlockType::DOCUMENT, std::forward<_Args>( aArgs )... ) );
            mCurrentBlock = mCurrentBlock->mChildren.back();
            mCurrentBlock->mDepth = d + 1;
            // SE::Logging::Info("{} Created block {}", std::string(d+1, '>'), typeid(_Ty).name());
        }

        template <typename _Ty, typename... _Args>
        void AppendBlock( _Args... aArgs )
        {
            uint32_t d = mCurrentBlock->mDepth;
            mCurrentBlock->mChildren.push_back( New<_Ty>( mCurrentBlock, eBlockType::DOCUMENT, std::forward<_Args>( aArgs )... ) );
            mCurrentBlock->mChildren.back()->mDepth = d + 1;
            // SE::Logging::Info("{} Append block {}", std::string(d+1, '>'), typeid(_Ty).name());
        }

        struct image_info
        {
            ImTextureID texture_id;
            ImVec2      size;
            ImVec2      uv0;
            ImVec2      uv1;
            ImVec4      col_tint;
            ImVec4      col_border;
        };

        // use m_href to identify image
        virtual bool get_image( image_info &nfo ) const;

        virtual ImFont *get_font() const;
        virtual ImVec4  get_color() const;

        // url == m_href
        virtual void open_url() const;

        // returns true if the term has been processed
        virtual bool render_entity( const char *str, const char *str_end );

        // returns true if the term has been processed
        virtual bool check_html( const char *str, const char *str_end );

        // called when '\n' in source text where it is not semantically meaningful
        virtual void soft_break();

        // e==true : enter
        // e==false : leave
        virtual void html_div( const std::string &dclass, bool e );
        ////////////////////////////////////////////////////////////////////////////

        // current state
        std::string m_href; // empty if no link/image

        bool     m_is_underline     = false;
        bool     m_is_strikethrough = false;
        bool     m_is_em            = false;
        bool     m_is_strong        = false;
        bool     m_is_table_header  = false;
        bool     m_is_table_body    = false;
        bool     m_is_image         = false;
        bool     m_is_code          = false;
        unsigned m_hlevel           = 0; // 0 - no heading

        uint32_t mBlockNestingLevel = 0;
        void     LogBlockType( bool e, const char *str );

      private:
        float mLeftMargin  = 10.0f;
        float mRightMargin = 50.0f;
        float mTextWidth   = 500.0f;

        int text( MD_TEXTTYPE type, const char *str, const char *str_end );
        int block( MD_BLOCKTYPE type, void *d, bool e );
        int span( MD_SPANTYPE type, void *d, bool e );

        void render_text( const char *str, const char *str_end );

        void set_font( bool e );
        void set_color( bool e );
        void set_href( bool e, const MD_ATTRIBUTE &src );

        static void line( ImColor c, bool under );

        // table state
        int                m_table_next_column = 0;
        ImVec2             m_table_last_pos;
        std::vector<float> m_table_col_pos;
        std::vector<float> m_table_row_pos;

        // list state
        struct list_info
        {
            unsigned cur_ol;
            char     delim;
            bool     is_ol;
        };
        std::vector<list_info> m_list_stack;

        std::vector<std::string> m_div_stack;

        MD_PARSER m_md;
    };
} // namespace SE::Core