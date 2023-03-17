#include "MultiPulseEventTable.h"

#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>
#include <numeric>

#include "Core/Profiling/BlockTimer.h"
#include "Mono/MonoRuntime.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    UIMultiPulseEventTable::UIMultiPulseEventTable()a
        : UITable()
    {
        SetRowHeight( 20.0f );

        mPositionColumn = New<sFloat64Column>( "Position", 75.0f, "{:.3f} km", "N.a.N." );
        AddColumn( mPositionColumn );

        mLossColumn = New<sFloat64Column>( "Loss", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mLossColumn );

        mEstimatedLossColumn = New<sFloat64Column>( "Est. Loss", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mEstimatedLossColumn );

        mReflectanceColumn = New<sFloat64Column>( "Reflectance", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mReflectanceColumn );

        mWavelengthColumn = New<sFloat64Column>( "Wavelength", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mWavelengthColumn );

        mSubCursorAColumn = New<sFloat64Column>( "a", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mSubCursorAColumn );

        mCursorAColumn = New<sFloat64Column>( "A", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mCursorAColumn );

        mCursorBColumn = New<sFloat64Column>( "B", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mCursorBColumn );

        mSubCursorBColumn = New<sFloat64Column>( "b", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mSubCursorBColumn );

        mCurveLevelColumn = New<sFloat64Column>( "Level", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mCurveLevelColumn );

        mLossAtAColumn = New<sFloat64Column>( "Loss@A", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mLossAtAColumn );

        mLossAtBColumn = New<sFloat64Column>( "Loss@B", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mLossAtBColumn );

        mEstimatedCurveLevelColumn = New<sFloat64Column>( "Est. Level", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mEstimatedCurveLevelColumn );

        mEstimatedEndLevelColumn = New<sFloat64Column>( "Est. End Level", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mEstimatedEndLevelColumn );

        mEndNoiseLevelColumn = New<sFloat64Column>( "Est. Noise Level", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mEndNoiseLevelColumn );

        mPeakPulseWidth = New<sFloat64Column>( "Pulse width", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mPeakPulseWidth );

        mPeakPower = New<sFloat64Column>( "Peak power", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mPeakPower );

        mPeakPower = New<sFloat64Column>( "Peak SNR", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mPeakPower );
    }

    void UIMultiPulseEventTable::OnEventClicked( std::function<void( sMultiPulseEvent const & )> const &aOnRowClicked )
    {
        mOnElementClicked = aOnRowClicked;
    }

    void UIMultiPulseEventTable::SetData( std::vector<sMultiPulseEvent> const &aData )
    {
        // Clear();

        // mEventDataVector = std::move( aData );

        // auto StringJoin = []( std::vector<std::string> aList )
        // {
        //     return aList.empty() ? "N/A"
        //                          : std::accumulate( aList.begin(), aList.end(), std::string(),
        //                                             []( const std::string &a, const std::string &b ) -> std::string
        //                                             { return a + ( a.length() > 0 ? ", " : "" ) + b; } );
        // };

        // static auto &lBaseLinkElementClass  = MonoRuntime::GetClassType( "Metrino.Olm.BaseLinkElement" );
        // static auto &lOlmPhysicalEventClass = MonoRuntime::GetClassType( "Metrino.Olm.OlmPhysicalEvent" );
        // static auto &lOlmAttributeClass     = MonoRuntime::GetClassType( "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute" );

        // for( auto const &lE : mEventDataVector )
        // {
        //     if( lE.mLinkIndex % 2 )
        //         mRowBackgroundColor.push_back( IM_COL32( 2, 2, 2, 255 ) );
        //     else
        //         mRowBackgroundColor.push_back( IM_COL32( 9, 9, 9, 255 ) );

        //     auto lLinkElement   = MonoScriptInstance( &lBaseLinkElementClass, lBaseLinkElementClass.Class(), lE.mLinkElement );
        //     auto lPhysicalEvent = MonoScriptInstance( &lOlmPhysicalEventClass, lOlmPhysicalEventClass.Class(), lE.mPhysicalEvent );
        //     auto lAttributes    = MonoScriptInstance( &lOlmAttributeClass, lOlmAttributeClass.Class(), lE.mAttributes );

        //     auto lOtdrPhysicalEvent = lPhysicalEvent.GetPropertyValue( "PhysicalEvent", "Metrino.Otdr.PhysicalEvent" );

        //     std::string lSplitterSource = lLinkElement.GetPropertyValue<bool>( "TwoByNSplitter" ) ? "2" : "1";
        //     if( lLinkElement.GetPropertyValue<eLinkElementType>( "Type" ) == eLinkElementType::Splitter )
        //     {
        //         auto lSplitterConfiguration =
        //             fmt::format( "{} ({}\xC3\x97{})", ToString( lLinkElement.GetPropertyValue<eLinkElementType>( "Type" ) ),
        //                          lSplitterSource, lLinkElement.GetPropertyValue<int>( "SplitterRatio" ) );
        //         mType->mData.push_back( lSplitterConfiguration );
        //     }
        //     else
        //     {
        //         mType->mData.push_back( ToString( lLinkElement.GetPropertyValue<eLinkElementType>( "Type" ) ) );
        //     }
        //     mStatus->mData.push_back( StringJoin( LinkStatusToString( lLinkElement.GetPropertyValue<int>( "Status" ) ) ) );

        //     mDiagnosicCount->mData.push_back( lE.mDiagnosicCount );
        //     mWavelength->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "Wavelength" ) * 1e9 );
        //     mPositionColumn->mData.push_back( lLinkElement.GetPropertyValue<double>( "Position" ) * 0.001f );
        //     mLoss->mData.push_back( lOtdrPhysicalEvent->GetPropertyValue<double>( "Loss" ) );

        //     if( lE.mLossPassFail == ePassFail::PASS )
        //         mLoss->mForegroundColor.emplace_back( IM_COL32( 0, 255, 0, 200 ) );
        //     else if( lE.mLossPassFail == ePassFail::FAIL )
        //         mLoss->mForegroundColor.emplace_back( IM_COL32( 255, 0, 0, 200 ) );
        //     else
        //         mLoss->mForegroundColor.emplace_back( IM_COL32( 0, 0, 0, 0 ) );

        //     mReflectance->mData.push_back( lOtdrPhysicalEvent->GetPropertyValue<double>( "Reflectance" ) );
        //     if( lE.mReflectancePassFail == ePassFail::PASS )
        //         mReflectance->mForegroundColor.emplace_back( IM_COL32( 0, 255, 0, 200 ) );
        //     else if( lE.mReflectancePassFail == ePassFail::FAIL )
        //         mReflectance->mForegroundColor.emplace_back( IM_COL32( 255, 0, 0, 200 ) );
        //     else
        //         mReflectance->mForegroundColor.emplace_back( IM_COL32( 0, 0, 0, 0 ) );

        //     mCurveLevelColumn->mData.push_back( lOtdrPhysicalEvent->GetPropertyValue<double>( "CurveLevel" ) );
        //     mEventType->mData.push_back( ToString( lOtdrPhysicalEvent->GetPropertyValue<eEventType>( "Type" ) ) );
        //     mEventStatus->mData.push_back( StringJoin( ToString( lOtdrPhysicalEvent->GetPropertyValue<int>( "Status" ) ) ) );
        //     mReflectanceType->mData.push_back(
        //         ToString( lOtdrPhysicalEvent->GetPropertyValue<eReflectanceType>( "ReflectanceType" ) ) );

        //     auto lEventSpanStart = lOtdrPhysicalEvent->GetPropertyValue<double>( "CursorA" ) * 0.001f;
        //     auto lEventSpanEnd   = lOtdrPhysicalEvent->GetPropertyValue<double>( "CursorB" ) * 0.001f;
        //     auto lEventRange     = fmt::format( "[{:.4f}, {:.4f}] km", lEventSpanStart, lEventSpanEnd );
        //     mEventSpan->mData.push_back( lEventRange );

        //     mPositionTolerance->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "PositionTolerance" ) );

        //     mLossError->mData.push_back( lAttributes.GetPropertyValue<double>( "LossError" ) );
        // }
    }

    void UIMultiPulseEventTable::Clear()
    {
        mPositionColumn.Clear();
        mLossColumn.Clear();
        mEstimatedLossColumn.Clear();
        mReflectanceColumn.Clear();
        mWavelengthColumn.Clear();
        mCursorAColumn.Clear();
        mCursorBColumn.Clear();
        mSubCursorAColumn.Clear();
        mSubCursorBColumn.Clear();
        mCurveLevelColumn.Clear();
        mLossAtAColumn.Clear();
        mLossAtBColumn.Clear();
        mEstimatedCurveLevelColumn.Clear();
        mEstimatedEndLevelColumn.Clear();
        mEndNoiseLevelColumn.Clear();
        mPeakPulseWidth.Clear();
        mPeakPower.Clear();
        mPeakSNR.Clear();
    }
} // namespace SE::OtdrEditor