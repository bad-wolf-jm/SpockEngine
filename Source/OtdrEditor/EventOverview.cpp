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
        mRbsTrace = New<UIPropertyValue>( "Pulse:" );
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

        if (!*aRbsData)
        {
            mRbsTrace->SetValue( "N/A" );
            mRbsNoise->SetValue( "N/A" );
            mRbsSaturation->SetValue( "N/A" );
            mLsaRange->SetValue( "N/A" );
            mLsaSlope->SetValue( "N/A" );
            mLsaOffset->SetValue( "N/A" );
            mLsaMean->SetValue( "N/A" );
            mLsaSlopeError->SetValue( "N/A" );
            mLsaLinear->SetValue( "N/A" );
            mLsaError->SetValue( "N/A" );
            mLsaCrossings->SetValue( "N/A" );

            return;
        }

        auto lTrace = aRbsData->GetPropertyValue( "Trace", "Metrino.Otdr.SinglePulseTrace" );

        std::string lFormat;
        if( lTrace && *lTrace )
        {
            std::string lPulse      = fmt::format( "{:.3f} ns", lTrace->GetPropertyValue<double>( "Pulse" ) * 1e9 );
            std::string lPulseWidth = fmt::format( "{:.3f} m", lTrace->GetPropertyValue<double>( "PulseWidth" ) );
            lFormat                 = fmt::format( "{} - {}", lPulse, lPulseWidth );
        }
        else
        {
            lFormat = "N/A";
        }
        mRbsTrace->SetValue( lFormat );

        std::string lRbsNoise = fmt::format( "{:.2f} dB", aRbsData->GetPropertyValue<double>( "NoiseLevel" ) );
        mRbsNoise->SetValue( lRbsNoise );

        std::string lRbsSaturation = fmt::format( "{:.2f} dB", aRbsData->GetPropertyValue<double>( "SaturationLevel" ) );
        mRbsSaturation->SetValue( lRbsSaturation );

        auto lLsaData = aRbsData->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );

        std::string lLsaRange;
        std::string lLsaSlope;
        std::string lLsaOffset;
        std::string lLsaMean;
        std::string lLsaSlopeError;
        std::string lLsaLinear;
        std::string lLsaError;
        std::string lLsaCrossings;

        if( lLsaData && *lLsaData )
        {
            lLsaRange      = fmt::format( "[{:.5f}, {:.5f}] km", lLsaData->GetPropertyValue<double>( "StartPosition" ) * 0.001,
                                          lLsaData->GetPropertyValue<double>( "EndPosition" ) * 0.001 );
            lLsaSlope      = fmt::format( "{:.3f} dB/km", lLsaData->GetPropertyValue<double>( "Slope" ) * 1000 );
            lLsaOffset     = fmt::format( "{:.3f} dB", lLsaData->GetPropertyValue<double>( "Offset" ) );
            lLsaMean       = fmt::format( "{:.3f} dB", lLsaData->GetPropertyValue<double>( "Mean" ) );
            lLsaSlopeError = fmt::format( "{:.3f} dB/km", lLsaData->GetPropertyValue<double>( "SlopeError" ) * 1000 );
            lLsaLinear     = fmt::format( "{}", lLsaData->GetPropertyValue<double>( "FitOnLinearData" ) );
            lLsaError      = fmt::format( "{:.3f} dB", lLsaData->GetPropertyValue<double>( "RmsError" ) );
            lLsaCrossings  = fmt::format( "{}", lLsaData->GetPropertyValue<int>( "Crossings" ) );
        }
        else
        {
            lLsaRange      = "N/A";
            lLsaSlope      = "N/A";
            lLsaOffset     = "N/A";
            lLsaMean       = "N/A";
            lLsaSlopeError = "N/A";
            lLsaLinear     = "N/A";
            lLsaError      = "N/A";
            lLsaCrossings  = "N/A";
        }

        mLsaRange->SetValue( lLsaRange );
        mLsaSlope->SetValue( lLsaSlope );
        mLsaOffset->SetValue( lLsaOffset );
        mLsaMean->SetValue( lLsaMean );
        mLsaSlopeError->SetValue( lLsaSlopeError );
        mLsaLinear->SetValue( lLsaLinear );
        mLsaError->SetValue( lLsaError );
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

        // Overview
        std::string lWavelength = fmt::format( "{:.0f} nm", aPhysicalEvent->GetPropertyValue<double>( "Wavelength" ) * 1e9 );
        mWavelength->SetValue( lWavelength );

        auto lDetectionTrace = aAttributes->GetPropertyValue( "DetectionTrace", "Metrino.Otdr.SinglePulseTrace" );

        if( lDetectionTrace && *lDetectionTrace )
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


        auto lOtdrPhysicalEvent = aPhysicalEvent->GetPropertyValue( "PhysicalEvent", "Metrino.Otdr.PhysicalEvent" );
        std::string lCurveLevel;
        std::string lNoiseLevel;
        if (lOtdrPhysicalEvent && *lOtdrPhysicalEvent)
        {
            lCurveLevel = fmt::format( "{:.2f} dB", lOtdrPhysicalEvent->GetPropertyValue<double>( "CurveLevel" ) );
            lNoiseLevel = fmt::format( "{:.2f} dB", lOtdrPhysicalEvent->GetPropertyValue<double>( "LocalNoise" ) );
        }
        else
        {
            lCurveLevel = "N/A";
            lNoiseLevel = "N/A";
        }
        mCurveLevel->SetValue( lCurveLevel );
        mEndOrNoiseLevel->SetValue( lNoiseLevel );

        std::string lEstimatedCurveLevel = fmt::format( "{:.2f} dB", aAttributes->GetPropertyValue<double>( "EstimatedCurveLevel" ) );
        mEstimatedCurveLevel->SetValue( lEstimatedCurveLevel );

        std::string lEstimatedEndLevel = fmt::format( "{:.2f} dB", aAttributes->GetPropertyValue<double>( "EstimatedEndLevel" ) );
        mEstimatedEndLevel->SetValue( lEstimatedEndLevel );

        std::string lEstimatedLoss = fmt::format( "{:.2f} dB", aAttributes->GetPropertyValue<double>( "EstimatedLoss" ) );
        mEstimatedLoss->SetValue( lEstimatedLoss );

        std::string lPeakSNR = fmt::format( "{:.2f} dB", aAttributes->GetPropertyValue<double>( "PeakSnr" ) );
        mPeakSNR->SetValue( lPeakSNR );

        auto lPeakTrace = aAttributes->GetPropertyValue( "PeakTrace", "Metrino.Otdr.SinglePulseTrace" );

        std::string lFormat;
        if( lPeakTrace && *lPeakTrace )
        {
            std::string lPeakPulse      = fmt::format( "{:.3f} ns", lPeakTrace->GetPropertyValue<double>( "Pulse" ) * 1e9 );
            std::string lPeakPulseWidth = fmt::format( "{:.3f} m", lPeakTrace->GetPropertyValue<double>( "PulseWidth" ) );
            lFormat                     = fmt::format( "{} - {}", lPeakPulse, lPeakPulseWidth );
        }
        else
        {
            lFormat = "N/A";
        }

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