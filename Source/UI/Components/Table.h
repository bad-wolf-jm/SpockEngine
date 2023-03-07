#pragma once

#include "Component.h"

namespace SE::Core
{
    struct sTableColumn
    {
        std::string mHeader;
        float       mInitialSize = 10.0f;

        virtual uint32_t Size()                               = 0;
        virtual void     Render( int aRow, ImVec2 aCellSize ) = 0;
    };

    struct sFloat64Column : public sTableColumn
    {
        std::string mFormat;
        std::string mNaNFormat;

        std::vector<double> mData;

        uint32_t Size();
        void     Render( int aRow, ImVec2 aCellSize );
    };

    struct sTableData
    {
        virtual int CountRows() = 0;

        virtual void *Get( int row, int column ) = 0;
    };

    class UITable : public UIComponent
    {
      public:
        UITable() = default;

        UITable( std::string const &aText );

        void AddColumn( Ref<sTableColumn> aColumn );
        void SetRowHeight( float aRowHeight );

      protected:
        std::vector<Ref<sTableColumn>> mColumns;
        Ref<sTableData>                mData;

      protected:
        float mRowHeight = 15.0f;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core