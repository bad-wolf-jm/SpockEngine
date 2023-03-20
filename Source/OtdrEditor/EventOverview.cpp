#include "EventOverview.h"

namespace SE::OtdrEditor
{
    SectionOverview::SectionOverview( std::string const &aTitle )
        : UIBoxLayout( eBoxLayoutOrientation::VERTICAL )
    {
        const float      lTitleHeight = 30.0f;
        const float      lItemHeight  = 20.0f;
        const math::vec4 lTitleBgColor{ 1.0f, 1.0f, 1.0f, 0.02f };

        mSectionTitle = New<UILabel>( aTitle );
        mSectionTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mSectionTitle->SetBackgroundColor( lTitleBgColor );
        Add( mSectionTitle.get(), lTitleHeight, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mSectionLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mSectionPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mSectionLayout->Add( nullptr, lTitleHeight, false, false );
        mSectionLayout->Add( mSectionPropertyLayout.get(), true, true );
        mRbsTrace = New<UIPropertyValue>( "Trace:" );
        mSectionPropertyLayout->Add( mRbsTrace.get(), lItemHeight, false, true );
        mRbsNoise = New<UIPropertyValue>( "Noise:" );
        mSectionPropertyLayout->Add( mRbsNoise.get(), lItemHeight, false, true );
        mRbsSaturation = New<UIPropertyValue>( "Saturation:" );
        mSectionPropertyLayout->Add( mRbsSaturation.get(), lItemHeight, false, true );
        mLsaRange = New<UIPropertyValue>( "LSA range:" );
        mSectionPropertyLayout->Add( mLsaRange.get(), lItemHeight, false, true );
        mLsaSlope = New<UIPropertyValue>( "LSA slope:" );
        mSectionPropertyLayout->Add( mLsaSlope.get(), lItemHeight, false, true );
        mLsaOffset = New<UIPropertyValue>( "LSA offset:" );
        mSectionPropertyLayout->Add( mLsaOffset.get(), lItemHeight, false, true );
        mLsaMean = New<UIPropertyValue>( "LSA mean:" );
        mSectionPropertyLayout->Add( mLsaMean.get(), lItemHeight, false, true );
        mLsaSlopeError = New<UIPropertyValue>( "LSA slope error:" );
        mSectionPropertyLayout->Add( mLsaSlopeError.get(), lItemHeight, false, true );
        mLsaLinear = New<UIPropertyValue>( "LSA linear:" );
        mSectionPropertyLayout->Add( mLsaLinear.get(), lItemHeight, false, true );
        mLsaError = New<UIPropertyValue>( "LSA error:" );
        mSectionPropertyLayout->Add( mLsaError.get(), lItemHeight, false, true );
        mLsaCrossings = New<UIPropertyValue>( "LSA crossings:" );
        mSectionPropertyLayout->Add( mLsaCrossings.get(), lItemHeight, false, true );
        mLsaSections = New<UIPropertyValue>( "LSA sections:" );
        mSectionPropertyLayout->Add( mLsaSections.get(), lItemHeight, false, true );
        Add( mSectionLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
    }

    void SectionOverview::SetData( Ref<MonoScriptInstance> aRbsData )
    {
        auto        lTrace      = aRbsData->GetPropertyValue( "Trace", "Metrino.Otdr.SinglePulseTrace" );
        std::string lPulse      = fmt::format( "{} ns", lTrace->GetPropertyValue<double>( "Pulse" ) * 1e9 );
        std::string lPulseWidth = fmt::format( "{:.3f} m", lTrace->GetPropertyValue<double>( "PulseWidth" ) );
        std::string lFormat     = fmt::format( "{} - {}", lPulse, lPulseWidth );
        mRbsTrace->SetValue( lFormat );

        std::string lRbsNoise = fmt::format( "{:.2f} dB", aRbsData->GetPropertyValue<double>( "NoiseLevel" ) );
        mRbsNoise->SetValue( lRbsNoise );

        std::string lRbsSaturation = fmt::format( "{:.2f} dB", aRbsData->GetPropertyValue<double>( "SaturationLevel" ) );
        mRbsSaturation->SetValue( lRbsSaturation );

        auto lLsaData = aRbsData->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );

        std::string lLsaRange = fmt::format( "[{:.3f}, {:.3f}]", lLsaData->GetPropertyValue<double>( "StartPosition" ),
                                             lLsaData->GetPropertyValue<double>( "EndPosition" ) );
        mLsaRange->SetValue( lLsaRange );

        std::string lLsaSlope = fmt::format( "{:.3f} dB/km", lLsaData->GetPropertyValue<double>( "Slope" ) * 1000 );
        mLsaSlope->SetValue( lLsaSlope );

        std::string lLsaOffset = fmt::format( "{:.3f} dB", lLsaData->GetPropertyValue<double>( "Offset" ) );
        mLsaOffset->SetValue( lLsaOffset );

        std::string lLsaMean = fmt::format( "{:.3f} dB", lLsaData->GetPropertyValue<double>( "Mean" ) );
        mLsaMean->SetValue( lLsaMean );

        std::string lLsaSlopeError = fmt::format( "{:.3f} dB/km", lLsaData->GetPropertyValue<double>( "SlopeError" ) );
        mLsaSlopeError->SetValue( lLsaSlopeError );

        std::string lLsaLinear = fmt::format( "{}", lLsaData->GetPropertyValue<double>( "FitOnLinearData" ) );
        mLsaLinear->SetValue( lLsaLinear );

        std::string lLsaError = fmt::format( "{:.3f} dB", lLsaData->GetPropertyValue<double>( "RmsError" ) );
        mLsaError->SetValue( lLsaError );

        std::string lLsaCrossings = fmt::format( "{}", lLsaData->GetPropertyValue<double>( "Crossings" ) );
        mLsaCrossings->SetValue( lLsaCrossings );
    }

    EventOverview::EventOverview()
        : UIBoxLayout( eBoxLayoutOrientation::VERTICAL )
    {
        const float      lTitleHeight = 30.0f;
        const float      lItemHeight  = 20.0f;
        const math::vec4 lTitleBgColor{ 1.0f, 1.0f, 1.0f, 0.02f };

        mEventOverview = New<UILabel>( "Event properties" );
        mEventOverview->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mEventOverview->SetBackgroundColor( lTitleBgColor );
        Add( mEventOverview.get(), lTitleHeight, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mEventOverviewLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mEventOverviewPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mEventOverviewLayout->Add( nullptr, 25.0f, false, false );
        mEventOverviewLayout->Add( mEventOverviewPropertyLayout.get(), true, true );
        mWavelength = New<UIPropertyValue>( "Wavelength:" );
        mEventOverviewPropertyLayout->Add( mWavelength.get(), lItemHeight, false, true );
        mDetectionTrace = New<UIPropertyValue>( "Detection trace:" );
        mEventOverviewPropertyLayout->Add( mDetectionTrace.get(), lItemHeight, false, true );
        mPositionTolerance = New<UIPropertyValue>( "Position tolerance:" );
        mEventOverviewPropertyLayout->Add( mPositionTolerance.get(), lItemHeight, false, true );
        mCurveLevel = New<UIPropertyValue>( "Curve level:" );
        mEventOverviewPropertyLayout->Add( mCurveLevel.get(), lItemHeight, false, true );
        mEndOrNoiseLevel = New<UIPropertyValue>( "Noise level:" );
        mEventOverviewPropertyLayout->Add( mEndOrNoiseLevel.get(), lItemHeight, false, true );
        mEstimatedCurveLevel = New<UIPropertyValue>( "Estimated curve level:" );
        mEventOverviewPropertyLayout->Add( mEstimatedCurveLevel.get(), lItemHeight, false, true );
        mEstimatedEndLevel = New<UIPropertyValue>( "Estimated end level:" );
        mEventOverviewPropertyLayout->Add( mEstimatedEndLevel.get(), lItemHeight, false, true );
        mEstimatedLoss = New<UIPropertyValue>( "Estimated loss:" );
        mEventOverviewPropertyLayout->Add( mEstimatedLoss.get(), lItemHeight, false, true );
        mPeakSNR = New<UIPropertyValue>( "Peak SNR:" );
        mEventOverviewPropertyLayout->Add( mPeakSNR.get(), lItemHeight, false, true );
        mPeakTrace = New<UIPropertyValue>( "Peak trace:" );
        mEventOverviewPropertyLayout->Add( mPeakTrace.get(), lItemHeight, false, true );
        mPeakPosition = New<UIPropertyValue>( "Peak position:" );
        mEventOverviewPropertyLayout->Add( mPeakPosition.get(), lItemHeight, false, true );
        mPeakPower = New<UIPropertyValue>( "Peak power:" );
        mEventOverviewPropertyLayout->Add( mPeakPower.get(), lItemHeight, false, true );
        mPeakSaturation = New<UIPropertyValue>( "Peak saturation:" );
        mEventOverviewPropertyLayout->Add( mPeakSaturation.get(), lItemHeight, false, true );
        mFresnelCorrection = New<UIPropertyValue>( "Fresnel correction:" );
        mEventOverviewPropertyLayout->Add( mFresnelCorrection.get(), lItemHeight, false, true );
        mUseSinglePulse = New<UIPropertyValue>( "Use single-pulse measurement:" );
        mEventOverviewPropertyLayout->Add( mUseSinglePulse.get(), lItemHeight, false, true );
        Add( mEventOverviewLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mPreviousSectionLayout = New<SectionOverview>( "Previous section" );
        Add( mPreviousSectionLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mNextSectionLayout = New<SectionOverview>( "Next section" );
        Add( mNextSectionLayout.get(), true, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
    }

    void EventOverview::SetData( Ref<MonoScriptInstance> aPhysicalEvent, Ref<MonoScriptInstance> aAttributes )
    {
        auto lOtdrPhysicalEvent = aPhysicalEvent->GetPropertyValue( "PhysicalEvent", "Metrino.Otdr.PhysicalEvent" );

        // Overview
        std::string lWavelength = fmt::format( "{:.0f} nm", aPhysicalEvent->GetPropertyValue<double>( "Wavelength" ) * 1e9 );
        mWavelength->SetValue( lWavelength );

        auto lDetectionTrace = aAttributes->GetPropertyValue( "DetectionTrace", "Metrino.Otdr.SinglePulseTrace" );

        if( lDetectionTrace )
        {
            std::string lDetectionTracePeakPulse =
                fmt::format( "{:.3f} ns", lDetectionTrace->GetPropertyValue<double>( "Pulse" ) * 1e9 );
            std::string lDetectionTracePeakPulseWidth =
                fmt::format( "{:.3f} m", lDetectionTrace->GetPropertyValue<double>( "PulseWidth" ) );
            std::string lDetectionTraceFormat = fmt::format( "{} - {}", lDetectionTracePeakPulse, lDetectionTracePeakPulseWidth );
            mDetectionTrace->SetValue( lDetectionTraceFormat );
        }
        else
        {
            mDetectionTrace->SetValue( "N/A" );
        }

        std::string lPositionTolerance = fmt::format( "{:.4f} m", aPhysicalEvent->GetPropertyValue<double>( "PositionTolerance" ) );
        mPositionTolerance->SetValue( lPositionTolerance );

        std::string lCurveLevel = fmt::format( "{:.2f} dB", lOtdrPhysicalEvent->GetPropertyValue<double>( "CurveLevel" ) );
        mCurveLevel->SetValue( lCurveLevel );

        std::string lNoiseLevel = fmt::format( "{:.2f} dB", lOtdrPhysicalEvent->GetPropertyValue<double>( "LocalNoise" ) );
        mEndOrNoiseLevel->SetValue( lNoiseLevel );

        std::string lEstimatedCurveLevel = fmt::format( "{:.2f} dB", aAttributes->GetPropertyValue<double>( "EstimatedCurveLevel" ) );
        mEstimatedCurveLevel->SetValue( lEstimatedCurveLevel );

        std::string lEstimatedEndLevel = fmt::format( "{:.2f} dB", aAttributes->GetPropertyValue<double>( "EstimatedEndLevel" ) );
        mEstimatedEndLevel->SetValue( lEstimatedEndLevel );

        std::string lEstimatedLoss = fmt::format( "{:.2f} dB", aAttributes->GetPropertyValue<double>( "EstimatedLoss" ) );
        mEstimatedLoss->SetValue( lEstimatedLoss );

        std::string lPeakSNR = fmt::format( "{:.2f} dB", aAttributes->GetPropertyValue<double>( "PeakSnr" ) );
        mPeakSNR->SetValue( lPeakSNR );

        auto        lPeakTrace      = aAttributes->GetPropertyValue( "PeakTrace", "Metrino.Otdr.SinglePulseTrace" );
        std::string lPeakPulse      = fmt::format( "{:.3f} ns", lPeakTrace->GetPropertyValue<double>( "Pulse" ) * 1e9 );
        std::string lPeakPulseWidth = fmt::format( "{:.3f} m", lPeakTrace->GetPropertyValue<double>( "PulseWidth" ) );
        std::string lFormat         = fmt::format( "{} - {}", lPeakPulse, lPeakPulseWidth );
        mPeakTrace->SetValue( lFormat );

        std::string lPeakPosition = fmt::format( "{:.2f} km", aAttributes->GetPropertyValue<double>( "PeakPosition" ) );
        mPeakPosition->SetValue( lPeakPosition );

        std::string lPeakPower = fmt::format( "{:.2f} dB", aAttributes->GetPropertyValue<double>( "PeakPower" ) );
        mPeakPower->SetValue( lPeakPower );

        std::string lPeakSaturation = fmt::format( "{:.2f} dB", aAttributes->GetPropertyValue<double>( "PeakSaturationLevel" ) );
        mPeakSaturation->SetValue( lPeakSaturation );

        std::string lFresnelCorrection = fmt::format( "{:.2f} dB", aAttributes->GetPropertyValue<double>( "FresnelCorrection" ) );
        mFresnelCorrection->SetValue( lFresnelCorrection );

        std::string lUseSinglePulse =
            fmt::format( "{}", aAttributes->GetPropertyValue<double>( "UseSinglePulseNextRbsForMeasurement" ) );
        mUseSinglePulse->SetValue( lUseSinglePulse );

        // Previous section
        auto lPreviousRbs = aAttributes->GetPropertyValue( "PreviousRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
        mPreviousSectionLayout->SetData( lPreviousRbs );
        // Next section
        auto lNextRbs = aAttributes->GetPropertyValue( "NextRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
        mNextSectionLayout->SetData( lNextRbs );
    }
} // namespace SE::OtdrEditor