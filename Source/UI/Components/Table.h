#pragma once

#include "Component.h"
#include <optional>

namespace SE::Core
{
    struct UITableColumn : public UIComponent
    {
        string_t mHeader;
        float    mInitialSize = 10.0f;

        std::vector<uint32_t>      mBackgroundColor;
        std::vector<uint32_t>      mForegroundColor;
        std::vector<UIComponent *> mToolTip;

        void PushStyles() {}
        void PopStyles() {}

        UITableColumn() = default;
        UITableColumn( string_t aHeader, float aInitialSize );

        ~UITableColumn() = default;

        virtual uint32_t Size()                               = 0;
        virtual void     Render( int aRow, ImVec2 aCellSize ) = 0;
        void             Clear();
    };

    struct UIFloat64Column : public UITableColumn
    {
        string_t mFormat;
        string_t mNaNFormat;

        std::vector<double> mData;

        UIFloat64Column() = default;
        UIFloat64Column( string_t aHeader, float aInitialSize, string_t aFormat, string_t aNaNFormat );

        ~UIFloat64Column() = default;

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

        uint32_t Size();
        void     Render( int aRow, ImVec2 aCellSize );
        void     Clear();
    };

    struct UIUint32Column : public UITableColumn
    {
        std::vector<uint32_t> mData;

        UIUint32Column() = default;
        UIUint32Column( string_t aHeader, float aInitialSize );

        ~UIUint32Column() = default;
        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

        uint32_t Size();
        void     Render( int aRow, ImVec2 aCellSize );
        void     Clear();
    };

    struct UIStringColumn : public UITableColumn
    {
        std::vector<string_t> mData;

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

        void AddColumn( Ref<UITableColumn> aColumn );
        void AddColumn( UITableColumn *aColumn );
        void SetRowHeight( float aRowHeight );

        void OnRowClicked( std::function<void( uint32_t )> const &aOnRowClicked );

        std::vector<uint32_t>           mRowBackgroundColor;
        std::optional<std::vector<int>> mDisplayedRowIndices;

      protected:
        std::vector<UITableColumn *>    mColumns;
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

      public:
        void *mOnRowClickDelegate       = nullptr;
        int   mOnRowClickDelegateHandle = -1;
    };
} // namespace SE::Core