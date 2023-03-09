#include "LinkElementTable.h"

#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>
#include <numeric>

#include "Core/Profiling/BlockTimer.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    UILinkElementTable::UILinkElementTable()
        : UITable()
    {
        SetRowHeight( 20.0f );

        mWavelength = New<sFloat64Column>( "Wavelength", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mWavelength );

        mType = New<sStringColumn>( "Type", 75.0f );
        AddColumn( mType );

        mStatus = New<sStringColumn>( "Status", 75.0f );
        AddColumn( mStatus );

        mPositionColumn = New<sFloat64Column>( "Position", 75.0f, "{:.3f} km", "N.a.N." );
        AddColumn( mPositionColumn );

        mDiagnosicCount = New<sFloat64Column>( "\xef\x86\x88", 75.0f, "{:.3f}", "N.a.N." );
        AddColumn( mDiagnosicCount );

        mLoss = New<sFloat64Column>( "Loss", 75.0f, "{:.3f} dB", "N.a.N." );
        AddColumn( mLoss );

        mReflectance = New<sFloat64Column>( "Reflectance", 75.0f, "{:.3f} dB", "N.a.N." );
        AddColumn( mReflectance );

        mCurveLevelColumn = New<sFloat64Column>( "Level", 75.0f, "{:.3f} dB", "N.a.N." );
        AddColumn( mCurveLevelColumn );

        mEventType = New<sStringColumn>( "mEventType", 75.0f );
        AddColumn( mEventType );

        mEventStatus = New<sStringColumn>( "mEventStatus", 75.0f );
        AddColumn( mEventStatus );

        mReflectanceType = New<sStringColumn>( "mReflectanceType", 75.0f );
        AddColumn( mReflectanceType );
    }

    void UILinkElementTable::SetData( std::vector<sLinkElement> const &aData )
    {
        mEventDataVector = std::move( aData );

        auto StringJoin = []( std::vector<std::string> aList )
        {
            return aList.empty() ? "N/A"
                                 : std::accumulate( aList.begin(), aList.end(), std::string(),
                                                    []( const std::string &a, const std::string &b ) -> std::string
                                                    { return a + ( a.length() > 0 ? ", " : "" ) + b; } );
        };

        mRowBackgroundColor.clear();
        for( auto const &lE : mEventDataVector ) mRowBackgroundColor.push_back( IM_COL32( 10, 10, 10, 255 * ( lE.mLinkIndex % 2 ) ) );

        mType->mData.clear();
        for( auto const &lE : mEventDataVector ) mType->mData.push_back( ToString( lE.mType ) );

        mStatus->mData.clear();
        for( auto const &lE : mEventDataVector ) mStatus->mData.push_back( StringJoin( LinkStatusToString( lE.mStatus ) ) );

        mDiagnosicCount->mData.clear();
        for( auto const &lE : mEventDataVector ) mDiagnosicCount->mData.push_back( lE.mDiagnosicCount );

        mWavelength->mData.clear();
        for( auto const &lE : mEventDataVector ) mWavelength->mData.push_back( lE.mWavelength * 1e9 );

        mPositionColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mPositionColumn->mData.push_back( lE.mPosition * 0.001f );

        mLoss->mData.clear();
        for( auto const &lE : mEventDataVector )
        {
            mLoss->mData.push_back( lE.mLoss );

            if( lE.mLossPassFail == ePassFail::PASS )
                mLoss->mForegroundColor.emplace_back( IM_COL32( 0, 255, 0, 200 ) );
            else if( lE.mLossPassFail == ePassFail::FAIL )
                mLoss->mForegroundColor.emplace_back( IM_COL32( 255, 0, 0, 200 ) );
            else
                mLoss->mForegroundColor.emplace_back( IM_COL32( 0, 0, 0, 0 ) );
        }

        mReflectance->mData.clear();
        for( auto const &lE : mEventDataVector )
        {
            mReflectance->mData.push_back( lE.mReflectance );
            if( lE.mReflectancePassFail == ePassFail::PASS )
                mReflectance->mForegroundColor.emplace_back( IM_COL32( 0, 255, 0, 200 ) );
            else if( lE.mReflectancePassFail == ePassFail::FAIL )
                mReflectance->mForegroundColor.emplace_back( IM_COL32( 255, 0, 0, 200 ) );
            else
                mReflectance->mForegroundColor.emplace_back( IM_COL32( 0, 0, 0, 0 ) );
        }

        mCurveLevelColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mCurveLevelColumn->mData.push_back( lE.mCurveLevel );

        mEventType->mData.clear();
        for( auto const &lE : mEventDataVector ) mEventType->mData.push_back( ToString( lE.mEventType ) );

        mEventStatus->mData.clear();
        for( auto const &lE : mEventDataVector ) mEventStatus->mData.push_back( StringJoin( ToString( lE.mEventStatus ) ) );

        mReflectanceType->mData.clear();
        for( auto const &lE : mEventDataVector ) mReflectanceType->mData.push_back( ToString( lE.mReflectanceType ) );
    }
} // namespace SE::OtdrEditor