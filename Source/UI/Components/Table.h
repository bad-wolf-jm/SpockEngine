#pragma once

#include "Component.h"

namespace SE::Core
{
    struct sTableColumn
    {
        std::string mHeader;
        float       mInitialSize = 10.0f;

        std::vector<uint32_t> mBackgroundColor;
        std::vector<uint32_t> mForegroundColor;

        sTableColumn() = default;
        sTableColumn( std::string aHeader, float aInitialSize );

        ~sTableColumn() = default;

        virtual uint32_t Size()                               = 0;
        virtual void     Render( int aRow, ImVec2 aCellSize ) = 0;
    };

    struct sFloat64Column : public sTableColumn
    {
        std::string mFormat;
        std::string mNaNFormat;

        std::vector<double> mData;

        sFloat64Column() = default;
        sFloat64Column( std::string aHeader, float aInitialSize, std::string aFormat, std::string aNaNFormat );
        ~sFloat64Column() = default;

        uint32_t Size();
        void     Render( int aRow, ImVec2 aCellSize );

      public:
        static void *UIFloat64Column_Create();
        static void *UIFloat64Column_CreateFull( void *aHeader, float aInitialSize, void *aFormat, void *aNaNFormat );
        static void  UIFloat64Column_Destroy( void *aSelf );
        static void  UIFloat64Column_Clear( void *aSelf );
        static void  UIFloat64Column_SetData( void *aSelf, void *aValue );
        static void  UIFloat64Column_SetDataWithForegroundColor( void *aSelf, void *aValue, void *aForegroundColor );
        static void  UIFloat64Column_SetDataWithForegroundAndBackgroundColor( void *aSelf, void *aValue, void *aForegroundColor,
                                                                              void *aBackroundColor );
    };

    struct sStringColumn : public sTableColumn
    {
        std::vector<std::string> mData;

        sStringColumn() = default;
        sStringColumn( std::string aHeader, float aInitialSize );
        ~sStringColumn() = default;

        uint32_t Size();
        void     Render( int aRow, ImVec2 aCellSize );

      public:
        static void *UIStringColumn_Create();
        static void *UIStringColumn_CreateFull( void *aHeader, float aInitialSize );
        static void  UIStringColumn_Destroy( void *aSelf );
        static void  UIStringColumn_Clear( void *aSelf );
        static void  UIStringColumn_SetData( void *aSelf, void *aValue );
        static void  UIStringColumn_SetDataWithForegroundColor( void *aSelf, void *aValue, void *aForegroundColor );
        static void  UIStringColumn_SetDataWithForegroundAndBackgroundColor( void *aSelf, void *aValue, void *aForegroundColor,
                                                                             void *aBackroundColor );
    };

    class UITable : public UIComponent
    {
      public:
        UITable() = default;

        // UITable( std::string const &aText );

        void AddColumn( Ref<sTableColumn> aColumn );
        void AddColumn( sTableColumn *aColumn );
        void SetRowHeight( float aRowHeight );

        void OnRowClicked( std::function<void( uint32_t )> const &aOnRowClicked );

        std::vector<uint32_t> mRowBackgroundColor;

      protected:
        std::vector<sTableColumn *>     mColumns;
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
        static void *UITable_Create();
        static void  UITable_Destroy( void *aSelf );
        static void  UITable_OnRowClicked( void *aSelf, void* aHandler );
        static void  UITable_AddColumn( void *aSelf, void *aColumn );
        static void  UITable_SetRowHeight( void *aSelf, float aRowHeight );
    };
} // namespace SE::Core