#include "LinkElementTable.h"

#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>
#include <numeric>

#include "Core/Profiling/BlockTimer.h"
#include "DotNet/Runtime.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    UILinkElementTable::UILinkElementTable()
        : UITable()
    {
        SetRowHeight( 20.0f );

        mIndex = New<sStringColumn>( "ID", 75.0f );
        AddColumn( mIndex );

        mWavelength = New<sFloat64Column>( "Wavelength", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mWavelength );

        mType = New<sStringColumn>( "Type", 75.0f );
        AddColumn( mType );

        mChanged = New<sStringColumn>( "Changed", 75.0f );
        AddColumn( mChanged );
        mHidden = New<sStringColumn>( "Hidden", 75.0f );
        AddColumn( mHidden );

        mStatus = New<sStringColumn>( "Status", 75.0f );
        AddColumn( mStatus );

        mPositionColumn = New<sFloat64Column>( "Position", 75.0f, "{:.4f} km", "N.a.N." );
        AddColumn( mPositionColumn );

        mDiagnosicCount = New<sFloat64Column>( "\xef\x86\x88", 75.0f, "{:.3f}", "N.a.N." );
        AddColumn( mDiagnosicCount );

        mLoss = New<sFloat64Column>( "Loss", 75.0f, "{:.3f} dB", "N.a.N." );
        AddColumn( mLoss );

        mReflectance = New<sFloat64Column>( "Reflectance", 75.0f, "{:.2f} dB", "N.a.N." );
        AddColumn( mReflectance );

        mCurveLevelColumn = New<sFloat64Column>( "Level", 75.0f, "{:.2f} dB", "N.a.N." );
        AddColumn( mCurveLevelColumn );

        mEventType = New<sStringColumn>( "mEventType", 75.0f );
        AddColumn( mEventType );

        mEventStatus = New<sStringColumn>( "mEventStatus", 75.0f );
        AddColumn( mEventStatus );

        mReflectanceType = New<sStringColumn>( "mReflectanceType", 75.0f );
        AddColumn( mReflectanceType );

        mEventSpan = New<sStringColumn>( "Range", 75.0f );
        AddColumn( mEventSpan );

        mPositionTolerance = New<sFloat64Column>( "Tolerance", 75.0f, "{:.3f} m", "N.a.N." );
        AddColumn( mPositionTolerance );

        mLossError = New<sFloat64Column>( "Loss Error", 75.0f, "{:.5f} dB", "N.a.N." );
        AddColumn( mLossError );

        mPeakPower = New<sFloat64Column>( "Peak power", 75.0f, "{:.2f} dB", "N.a.N." );
        AddColumn( mPeakPower );

        mComment = New<sStringColumn>( "Comment", 75.0f );
        AddColumn( mComment );

        UITable::OnRowClicked(
            [&]( uint32_t aRow )
            {
                if( mOnElementClicked ) mOnElementClicked( mEventDataVector[aRow] );
            } );
    }

    void UILinkElementTable::OnElementClicked( std::function<void( sLinkElement const & )> const &aOnRowClicked )
    {
        mOnElementClicked = aOnRowClicked;
    }

    void UILinkElementTable::SetData( std::vector<sLinkElement> const &aData )
    {
        Clear();

        mEventDataVector = std::move( aData );

        auto StringJoin = []( std::vector<std::string> aList )
        {
            return aList.empty() ? "N/A"
                                 : std::accumulate( aList.begin(), aList.end(), std::string(),
                                                    []( const std::string &a, const std::string &b ) -> std::string
                                                    { return a + ( a.length() > 0 ? ", " : "" ) + b; } );
        };

        static auto &lBaseLinkElementClass  = DotNetRuntime::GetClassType( "Metrino.Olm.BaseLinkElement" );
        static auto &lOlmPhysicalEventClass = DotNetRuntime::GetClassType( "Metrino.Olm.OlmPhysicalEvent" );
        static auto &lOlmAttributeClass     = DotNetRuntime::GetClassType( "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute" );

        for( auto const &lE : mEventDataVector )
        {
            mIndex->mData.push_back(
                fmt::format( "{} - {} - {} - {}", lE.mLinkIndex, lE.mSubLinkIndex, lE.mEventIndex, lE.mIsSubElement ) );
            if( lE.mLinkIndex % 2 )
                mRowBackgroundColor.push_back( IM_COL32( 2, 2, 2, 255 ) );
            else
                mRowBackgroundColor.push_back( IM_COL32( 9, 9, 9, 255 ) );

            auto lLinkElement   = DotNetInstance( &lBaseLinkElementClass, lBaseLinkElementClass.Class(), lE.mLinkElement );
            auto lPhysicalEvent = DotNetInstance( &lOlmPhysicalEventClass, lOlmPhysicalEventClass.Class(), lE.mPhysicalEvent );
            auto lAttributes    = DotNetInstance( &lOlmAttributeClass, lOlmAttributeClass.Class(), lE.mAttributes );

            auto lOtdrPhysicalEvent = lPhysicalEvent.GetPropertyValue( "PhysicalEvent", "Metrino.Otdr.PhysicalEvent" );

            std::string lSplitterSource = lLinkElement.GetPropertyValue<bool>( "TwoByNSplitter" ) ? "2" : "1";
            if( lLinkElement.GetPropertyValue<eLinkElementType>( "Type" ) == eLinkElementType::Splitter )
            {
                auto lSplitterConfiguration =
                    fmt::format( "{} ({}\xC3\x97{})", ToString( lLinkElement.GetPropertyValue<eLinkElementType>( "Type" ) ),
                                 lSplitterSource, lLinkElement.GetPropertyValue<int>( "SplitterRatio" ) );
                mType->mData.push_back( lSplitterConfiguration );
            }
            else
            {
                mType->mData.push_back( ToString( lLinkElement.GetPropertyValue<eLinkElementType>( "Type" ) ) );
            }
            mStatus->mData.push_back( StringJoin( LinkStatusToString( lLinkElement.GetPropertyValue<int>( "Status" ) ) ) );

            mChanged->mData.push_back(lLinkElement.GetPropertyValue<bool>( "Changed" ) ? "True" : "False");
            mHidden->mData.push_back(lLinkElement.GetPropertyValue<bool>( "Hidden" ) ? "True" : "False");

            auto lComment = lLinkElement.GetPropertyValue<MonoString*>( "Comment" );
            mComment->mData.push_back(DotNetRuntime::NewString(lComment));

            mDiagnosicCount->mData.push_back( lE.mDiagnosicCount );
            mWavelength->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "Wavelength" ) * 1e9 );
            mPositionColumn->mData.push_back( lLinkElement.GetPropertyValue<double>( "Position" ) * 0.001f );
            mLoss->mData.push_back( lOtdrPhysicalEvent->GetPropertyValue<double>( "Loss" ) );

            if( lE.mLossPassFail == ePassFail::PASS )
                mLoss->mForegroundColor.emplace_back( IM_COL32( 0, 255, 0, 200 ) );
            else if( lE.mLossPassFail == ePassFail::FAIL )
                mLoss->mForegroundColor.emplace_back( IM_COL32( 255, 0, 0, 200 ) );
            else
                mLoss->mForegroundColor.emplace_back( IM_COL32( 0, 0, 0, 0 ) );

            mReflectance->mData.push_back( lOtdrPhysicalEvent->GetPropertyValue<double>( "Reflectance" ) );
            if( lE.mReflectancePassFail == ePassFail::PASS )
                mReflectance->mForegroundColor.emplace_back( IM_COL32( 0, 255, 0, 200 ) );
            else if( lE.mReflectancePassFail == ePassFail::FAIL )
                mReflectance->mForegroundColor.emplace_back( IM_COL32( 255, 0, 0, 200 ) );
            else
                mReflectance->mForegroundColor.emplace_back( IM_COL32( 0, 0, 0, 0 ) );

            mCurveLevelColumn->mData.push_back( lOtdrPhysicalEvent->GetPropertyValue<double>( "CurveLevel" ) );
            mEventType->mData.push_back( ToString( lOtdrPhysicalEvent->GetPropertyValue<eEventType>( "Type" ) ) );
            mEventStatus->mData.push_back( StringJoin( ToString( lOtdrPhysicalEvent->GetPropertyValue<int>( "Status" ) ) ) );
            mReflectanceType->mData.push_back(
                ToString( lOtdrPhysicalEvent->GetPropertyValue<eReflectanceType>( "ReflectanceType" ) ) );

            auto lEventSpanStart = lOtdrPhysicalEvent->GetPropertyValue<double>( "CursorA" ) * 0.001f;
            auto lEventSpanEnd   = lOtdrPhysicalEvent->GetPropertyValue<double>( "CursorB" ) * 0.001f;
            auto lEventRange     = fmt::format( "[{:.4f}, {:.4f}] km", lEventSpanStart, lEventSpanEnd );
            mEventSpan->mData.push_back( lEventRange );

            mPositionTolerance->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "PositionTolerance" ) );

            if( lAttributes )
            {
                mLossError->mData.push_back( lAttributes.GetPropertyValue<double>( "LossError" ) );
                mPeakPower->mData.push_back( lAttributes.GetPropertyValue<double>( "PeakPower" ) );
            }
            else
            {
                mLossError->mData.push_back( nan("") );
                mPeakPower->mData.push_back( nan("") );
            }
        }
    }

    void UILinkElementTable::Clear()
    {
        mRowBackgroundColor.clear();
        mType->mData.clear();
        mStatus->mData.clear();
        mDiagnosicCount->mData.clear();
        mWavelength->mData.clear();
        mPositionColumn->mData.clear();
        mLoss->mData.clear();
        mReflectance->mData.clear();
        mCurveLevelColumn->mData.clear();
        mEventType->mData.clear();
        mEventStatus->mData.clear();
        mReflectanceType->mData.clear();
        mEventSpan->mData.clear();
        mPositionTolerance->mData.clear();
        mLossError->mData.clear();
    }

    std::vector<sLinkElement> UILinkElementTable::GetElementsByIndex( uint32_t aElementIndex )
    {
        std::vector<sLinkElement> lResult;

        for( auto const &x : mEventDataVector )
            if( x.mLinkIndex == aElementIndex ) lResult.push_back( x );

        return std::move( lResult );
    }
} // namespace SE::OtdrEditor