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

    UIMultiPulseEventTable::UIMultiPulseEventTable()
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

        mPeakSNR = New<sFloat64Column>( "Peak SNR", 75.0f, "{:.1f} nm", "N.a.N." );
        AddColumn( mPeakSNR );
    }

    void UIMultiPulseEventTable::OnEventClicked( std::function<void( sMultiPulseEvent const & )> const &aOnRowClicked )
    {
        mOnElementClicked = aOnRowClicked;
    }

    void UIMultiPulseEventTable::SetData( std::vector<sMultiPulseEvent> const &aData )
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

        static auto &lPhysicalEventClass = MonoRuntime::GetClassType( "Metrino.Otdr.PhysicalEvent" );
        static auto &lOlmAttributeClass  = MonoRuntime::GetClassType( "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute" );

        for( auto const &lE : mEventDataVector )
        {
            auto lPhysicalEvent = MonoScriptInstance( &lPhysicalEventClass, lPhysicalEventClass.Class(), lE.mPhysicalEvent );
            auto lAttributes    = lPhysicalEvent.GetPropertyValue( "Tag", "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute" );
            auto lPeakTrace     = lAttributes->GetPropertyValue( "PeakTrace", "Metrino.Otdr.SinglePulseTrace" );

            mPositionColumn->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "Position" ) * 0.001f );
            mLossColumn->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "Loss" ) );
            mEstimatedLossColumn->mData.push_back( lAttributes->GetPropertyValue<double>( "EstimatedLoss" ) );
            mReflectanceColumn->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "Reflectance" ) );

            if (*lPeakTrace)
            {
                mPeakPulseWidth->mData.push_back( lPeakTrace->GetPropertyValue<double>( "Pulse" ) * 1e9 );
                mWavelengthColumn->mData.push_back( lPeakTrace->GetPropertyValue<double>( "Wavelength" ) * 1e9 );
            }
            else
            {
                mPeakPulseWidth->mData.push_back( nan("") );
                mWavelengthColumn->mData.push_back( nan("") );
            }

            mCursorAColumn->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "CursorA" ) );
            mCursorBColumn->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "CursorB" ) );
            mSubCursorAColumn->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "SubCursorA" ) );
            mSubCursorBColumn->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "SubCursorB" ) );
            mCurveLevelColumn->mData.push_back( lPhysicalEvent.GetPropertyValue<double>( "CurveLevel" ) );
            mLossAtAColumn->mData.push_back( lAttributes->GetPropertyValue<double>( "LossAtA" ) );
            mLossAtBColumn->mData.push_back( lAttributes->GetPropertyValue<double>( "LossAtB" ) );
            mEstimatedCurveLevelColumn->mData.push_back( lAttributes->GetPropertyValue<double>( "EstimatedCurveLevel" ) );
            mEstimatedEndLevelColumn->mData.push_back( lAttributes->GetPropertyValue<double>( "EstimatedEndLevel" ) );
            mEndNoiseLevelColumn->mData.push_back( lAttributes->GetPropertyValue<double>( "EndNoiseLevel" ) );
            mPeakPower->mData.push_back( lAttributes->GetPropertyValue<double>( "PeakPower" ) );
            mPeakSNR->mData.push_back( lAttributes->GetPropertyValue<double>( "PeakSnr" ) );
        }
    }

    void UIMultiPulseEventTable::Clear()
    {
        mPositionColumn->mData.clear();
        mLossColumn->mData.clear();
        mEstimatedLossColumn->mData.clear();
        mReflectanceColumn->mData.clear();
        mWavelengthColumn->mData.clear();
        mCursorAColumn->mData.clear();
        mCursorBColumn->mData.clear();
        mSubCursorAColumn->mData.clear();
        mSubCursorBColumn->mData.clear();
        mCurveLevelColumn->mData.clear();
        mLossAtAColumn->mData.clear();
        mLossAtBColumn->mData.clear();
        mEstimatedCurveLevelColumn->mData.clear();
        mEstimatedEndLevelColumn->mData.clear();
        mEndNoiseLevelColumn->mData.clear();
        mPeakPulseWidth->mData.clear();
        mPeakPower->mData.clear();
        mPeakSNR->mData.clear();
    }
} // namespace SE::OtdrEditor