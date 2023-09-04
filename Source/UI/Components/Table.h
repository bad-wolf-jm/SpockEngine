#pragma once

#include "Component.h"
#include <optional>

namespace SE::Core
{
    struct sTableColumn
    {
        string_t mHeader;
        float       mInitialSize = 10.0f;

        vector_t<uint32_t>      mBackgroundColor;
        vector_t<uint32_t>      mForegroundColor;
        vector_t<UIComponent *> mToolTip;

        sTableColumn() = default;
        sTableColumn( string_t aHeader, float aInitialSize );

        ~sTableColumn() = default;

        virtual uint32_t Size()                               = 0;
        virtual void     Render( int aRow, ImVec2 aCellSize ) = 0;
        void             Clear();

        static void UITableColumn_SetTooltip( void *aSelf, void *aTooptip );
        static void UITableColumn_SetForegroundColor( void *aSelf, void *aForegroundColor );
        static void UITableColumn_SetBackgroundColor( void *aSelf, void *aBackroundColor );
    };

    struct sFloat64Column : public sTableColumn
    {
        string_t mFormat;
        string_t mNaNFormat;

        vector_t<double> mData;

        sFloat64Column() = default;
        sFloat64Column( string_t aHeader, float aInitialSize, string_t aFormat, string_t aNaNFormat );
        ~sFloat64Column() = default;

        uint32_t Size();
        void     Render( int aRow, ImVec2 aCellSize );
        void     Clear();

      public:
        static void *UIFloat64Column_Create();
        static void *UIFloat64Column_CreateFull( void *aHeader, float aInitialSize, void *aFormat, void *aNaNFormat );
        static void  UIFloat64Column_Destroy( void *aSelf );
        static void  UIFloat64Column_Clear( void *aSelf );
        static void  UIFloat64Column_SetData( void *aSelf, void *aValue );
    };

    struct sUint32Column : public sTableColumn
    {
        vector_t<uint32_t> mData;

        sUint32Column() = default;
        sUint32Column( string_t aHeader, float aInitialSize );
        ~sUint32Column() = default;

        uint32_t Size();
        void     Render( int aRow, ImVec2 aCellSize );
        void     Clear();

      public:
        static void *UIUint32Column_Create();
        static void *UIUint32Column_CreateFull( void *aHeader, float aInitialSize );
        static void  UIUint32Column_Destroy( void *aSelf );
        static void  UIUint32Column_Clear( void *aSelf );
        static void  UIUint32Column_SetData( void *aSelf, void *aValue );
    };

    struct sStringColumn : public sTableColumn
    {
        vector_t<string_t> mData;

        sStringColumn() = default;
        sStringColumn( string_t aHeader, float aInitialSize );
        ~sStringColumn() = default;

        uint32_t Size();
        void     Render( int aRow, ImVec2 aCellSize );
        void     Clear();

      public:
        static void *UIStringColumn_Create();
        static void *UIStringColumn_CreateFull( void *aHeader, float aInitialSize );
        static void  UIStringColumn_Destroy( void *aSelf );
        static void  UIStringColumn_Clear( void *aSelf );
        static void  UIStringColumn_SetData( void *aSelf, void *aValue );
    };

    class UITable : public UIComponent
    {
      public:
        UITable() = default;

        void AddColumn( ref_t<sTableColumn> aColumn );
        void AddColumn( sTableColumn *aColumn );
        void SetRowHeight( float aRowHeight );

        void OnRowClicked( std::function<void( uint32_t )> const &aOnRowClicked );

        vector_t<uint32_t>           mRowBackgroundColor;
        std::optional<vector_t<int>> mDisplayedRowIndices;

      protected:
        vector_t<sTableColumn *>     mColumns;
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

      private:
        void *mOnRowClickDelegate       = nullptr;
        int   mOnRowClickDelegateHandle = -1;

      public:
        static void *UITable_Create();
        static void  UITable_Destroy( void *aSelf );
        static void  UITable_OnRowClicked( void *aSelf, void *aHandler );
        static void  UITable_AddColumn( void *aSelf, void *aColumn );
        static void  UITable_SetRowHeight( void *aSelf, float aRowHeight );
        static void  UITable_SetRowBackgroundColor( void *aSelf, void *aColors );
        static void  UITable_SetDisplayedRowIndices( void *aSelf, void *aIndices );
        static void  UITable_ClearRowBackgroundColor( void *aSelf );
    };
} // namespace SE::Core