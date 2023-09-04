#pragma once

#include "Component.h"
#include <optional>

namespace SE::Core
{
    struct UITableColumn : public UIComponent
    {
        string_t mHeader;
        float    mInitialSize = 10.0f;

        vector_t<uint32_t>      mBackgroundColor;
        vector_t<uint32_t>      mForegroundColor;
        vector_t<UIComponent *> mToolTip;

        void PushStyles()
        {
        }
        void PopStyles()
        {
        }

        UITableColumn() = default;
        UITableColumn( string_t aHeader, float aInitialSize );

        ~UITableColumn() = default;

        virtual uint32_t Size()                               = 0;
        virtual void     Render( int aRow, ImVec2 aCellSize ) = 0;
        void             Clear();
    };

    struct UIStringColumn : public UITableColumn
    {
        vector_t<string_t> mData;

        UIStringColumn() = default;
        UIStringColumn( string_t aHeader, float aInitialSize );

        ~UIStringColumn() = default;
        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

        uint32_t Size();
        void     Render( int aRow, ImVec2 aCellSize );
        void     Clear();
    };

    class UITable : public UIComponent
    {
      public:
        UITable() = default;

        void AddColumn( ref_t<UITableColumn> aColumn );
        void AddColumn( UITableColumn *aColumn );
        void SetRowHeight( float aRowHeight );

        void OnRowClicked( std::function<void( uint32_t )> const &aOnRowClicked );

        vector_t<uint32_t>           mRowBackgroundColor;
        std::optional<vector_t<int>> mDisplayedRowIndices;

      protected:
        vector_t<UITableColumn *>       mColumns;
        std::function<void( uint32_t )> mOnRowClicked;

      protected:
        float   mRowHeight   = 15.0f;
        int32_t mHoveredRow  = -1;
        int32_t mSelectedRow = -1;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
        void   DrawTableRows( ImGuiTable *lThisTable, uint32_t lRowCount, int aRowStart, int aRowEnd );
        void   DrawTableCell( ImGuiTable *lThisTable, UITableColumn *lColumnData, int lColumn, int lRow );

      public:
        void *mOnRowClickDelegate       = nullptr;
        int   mOnRowClickDelegateHandle = -1;
    };
} // namespace SE::Core