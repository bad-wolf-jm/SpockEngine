#include "AcquisitionData.h"
#include "DotNet/Runtime.h"
namespace SE::OtdrEditor
{
    AcquisitionData::AcquisitionData()
        : UIBoxLayout( eBoxLayoutOrientation::VERTICAL )
    {
        const float lTitleHeight = 30.0f;
        const float lItemHeight  = 20.0f;
        const math::vec4 lTitleBgColor{ 1.0f, 1.0f, 1.0f, 0.02f };

        mSectionTitle = New<UILabel>( "Acquisition data" );
        mSectionTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mSectionTitle->SetBackgroundColor( lTitleBgColor );
        Add( mSectionTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mSectionLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mSectionPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mSectionLayout->Add( nullptr, 25.0f, false, false );
        mSectionLayout->Add( mSectionPropertyLayout.get(), true, true );

        mWavelength = New<UIPropertyValue>( "Wavelength:" );
        mSectionPropertyLayout->Add( mWavelength.get(), lItemHeight, false, true );
        mPulse = New<UIPropertyValue>( "Pulse:" );
        mSectionPropertyLayout->Add( mPulse.get(), lItemHeight, false, true );
        mRange = New<UIPropertyValue>( "Range:" );
        mSectionPropertyLayout->Add( mRange.get(), lItemHeight, false, true );
        mStitchRange = New<UIPropertyValue>( "Stitch range:" );
        mSectionPropertyLayout->Add( mStitchRange.get(), lItemHeight, false, true );
        mLocalError = New<UIPropertyValue>( "Local error:" );
        mSectionPropertyLayout->Add( mLocalError.get(), lItemHeight, false, true );
        mAverages = New<UIPropertyValue>( "Averages:" );
        mSectionPropertyLayout->Add( mAverages.get(), lItemHeight, false, true );
        mDecimationsPhases = New<UIPropertyValue>( "Decimation / Phase:" );
        mSectionPropertyLayout->Add( mDecimationsPhases.get(), lItemHeight, false, true );
        mDeltaT = New<UIPropertyValue>( "\xCE\x94T:" );
        mSectionPropertyLayout->Add( mDeltaT.get(), lItemHeight, false, true );
        mTimeToOutput = New<UIPropertyValue>( "Time to output:" );
        mSectionPropertyLayout->Add( mTimeToOutput.get(), lItemHeight, false, true );
        mTimeLASToOutput = New<UIPropertyValue>( "Time LAS to output:" );
        mSectionPropertyLayout->Add( mTimeLASToOutput.get(), lItemHeight, false, true );
        mTTOInternalSampling = New<UIPropertyValue>( "TTO internal sam:" );
        mSectionPropertyLayout->Add( mTTOInternalSampling.get(), lItemHeight, false, true );
        mInternalReflectance = New<UIPropertyValue>( "Internal reflectance:" );
        mSectionPropertyLayout->Add( mInternalReflectance.get(), lItemHeight, false, true );
        mSamplingDelay = New<UIPropertyValue>( "Repetition period:" );
        mSectionPropertyLayout->Add( mSamplingDelay.get(), lItemHeight, false, true );
        mRepetitionPeriod = New<UIPropertyValue>( "Repetition period:" );
        mSectionPropertyLayout->Add( mRepetitionPeriod.get(), lItemHeight, false, true );
        mAcquisitionTime = New<UIPropertyValue>( "Acquisition time:" );
        mSectionPropertyLayout->Add( mAcquisitionTime.get(), lItemHeight, false, true );
        mTzCode = New<UIPropertyValue>( "TZ Code:" );
        mSectionPropertyLayout->Add( mTzCode.get(), lItemHeight, false, true );
        mBandwidth = New<UIPropertyValue>( "Bandwidth:" );
        mSectionPropertyLayout->Add( mBandwidth.get(), lItemHeight, false, true );
        mAnalogGain = New<UIPropertyValue>( "Analog gain:" );
        mSectionPropertyLayout->Add( mAnalogGain.get(), lItemHeight, false, true );
        mApdGain = New<UIPropertyValue>( "Apd gain:" );
        mSectionPropertyLayout->Add( mApdGain.get(), lItemHeight, false, true );
        mStitchGain = New<UIPropertyValue>( "Stitch gain:" );
        mSectionPropertyLayout->Add( mStitchGain.get(), lItemHeight, false, true );
        mFresnelCorrection = New<UIPropertyValue>( "Fresnel correction:" );
        mSectionPropertyLayout->Add( mFresnelCorrection.get(), lItemHeight, false, true );
        mNormalizationFactor = New<UIPropertyValue>( "Normalization factor:" );
        mSectionPropertyLayout->Add( mNormalizationFactor.get(), lItemHeight, false, true );
        mHighBandwidthFilter = New<UIPropertyValue>( "High bandwidth filter:" );
        mSectionPropertyLayout->Add( mHighBandwidthFilter.get(), lItemHeight, false, true );
        mUnfilteredRmsNoise = New<UIPropertyValue>( "Unfiltered RMS noise:" );
        mSectionPropertyLayout->Add( mUnfilteredRmsNoise.get(), lItemHeight, false, true );
        mLowBandwidthFilterNoise = New<UIPropertyValue>( "Low bandwidth filter noise:" );
        mSectionPropertyLayout->Add( mLowBandwidthFilterNoise.get(), lItemHeight, false, true );
        mHighBandwidthFiltedNoise = New<UIPropertyValue>( "High bandwidth filter noise:" );
        mSectionPropertyLayout->Add( mHighBandwidthFiltedNoise.get(), lItemHeight, false, true );
        mExpectedInjection = New<UIPropertyValue>( "Expected injections:" );
        mSectionPropertyLayout->Add( mExpectedInjection.get(), lItemHeight, false, true );
        mSaturationLevel = New<UIPropertyValue>( "Saturation level:" );
        mSectionPropertyLayout->Add( mSaturationLevel.get(), lItemHeight, false, true );
        mFresnelSaturation = New<UIPropertyValue>( "Fresnel saturationr:" );
        mSectionPropertyLayout->Add( mFresnelSaturation.get(), lItemHeight, false, true );
        mSpeckeNoise = New<UIPropertyValue>( "Speckle noise:" );
        mSectionPropertyLayout->Add( mSpeckeNoise.get(), lItemHeight, false, true );
        mSaturationMaskRatio = New<UIPropertyValue>( "Saturation Masking Ratio:" );
        mSectionPropertyLayout->Add( mSaturationMaskRatio.get(), lItemHeight, false, true );
        mSaturationMaskDuration = New<UIPropertyValue>( "Saturation Masking Duration:" );
        mSectionPropertyLayout->Add( mSaturationMaskDuration.get(), lItemHeight, false, true );
        mApdID = New<UIPropertyValue>( "APD Id:" );
        mSectionPropertyLayout->Add( mApdID.get(), lItemHeight, false, true );
        mRiseTime = New<UIPropertyValue>( "Rise time:" );
        mSectionPropertyLayout->Add( mRiseTime.get(), lItemHeight, false, true );
        mFallTime = New<UIPropertyValue>( "Fall time:" );
        mSectionPropertyLayout->Add( mFallTime.get(), lItemHeight, false, true );
        mPositionTolerance = New<UIPropertyValue>( "Position tolerance:" );
        mSectionPropertyLayout->Add( mPositionTolerance.get(), lItemHeight, false, true );
        mFiberCode = New<UIPropertyValue>( "Fiber code:" );
        mSectionPropertyLayout->Add( mFiberCode.get(), lItemHeight, false, true );
        mIOR = New<UIPropertyValue>( "IOR:" );
        mSectionPropertyLayout->Add( mIOR.get(), lItemHeight, false, true );
        mRBS = New<UIPropertyValue>( "RBS:" );
        mSectionPropertyLayout->Add( mRBS.get(), lItemHeight, false, true );
        mAttenuation = New<UIPropertyValue>( "Attenuation:" );
        mSectionPropertyLayout->Add( mAttenuation.get(), lItemHeight, false, true );
        mSpanOrl = New<UIPropertyValue>( "Span ORL:" );
        mSectionPropertyLayout->Add( mSpanOrl.get(), lItemHeight, false, true );
        mSequenceType = New<UIPropertyValue>( "Sequence type:" );
        mSectionPropertyLayout->Add( mSequenceType.get(), lItemHeight, false, true );

        Add( mSectionLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
    }

    void AcquisitionData::SetData( Ref<DotNetInstance> aSinglePulseTrace, Ref<DotNetInstance> aAcquisitionData,
                                   Ref<DotNetInstance> aFiberInfo )
    {
        mWavelength->SetValue( "{:.0f} nm", aSinglePulseTrace->GetPropertyValue<double>( "Wavelength" ) * 1e9 );

        std::string lPulse       = fmt::format( "{:.3} \xCE\xBCs", aSinglePulseTrace->GetPropertyValue<double>( "Pulse" ) * 1e6 );
        std::string lPulseWidth  = fmt::format( "{:.3f} m", aSinglePulseTrace->GetPropertyValue<double>( "PulseWidth" ) );
        std::string lPulseFormat = fmt::format( "{} - {}", lPulse, lPulseWidth );
        mPulse->SetValue( lPulseFormat );

        std::string lRange       = fmt::format( "{:.3f}", aAcquisitionData->GetPropertyValue<double>( "RangeStart" ) * 1e6 );
        std::string lRangeWidth  = fmt::format( "{:.3f}", aAcquisitionData->GetPropertyValue<double>( "RangeEnd" ) * 1e6 );
        std::string lRangeFormat = fmt::format( "[{}, {}] \xCE\xBCs", lRange, lRangeWidth );
        mRange->SetValue( lRangeFormat );

        std::string lStitchRange =
            fmt::format( "{:.3f}", aAcquisitionData->GetPropertyValue<double>( "StitchComputationStart" ) * 1e6 );
        std::string lStitchRangeWidth =
            fmt::format( "{:.3f}", aAcquisitionData->GetPropertyValue<double>( "StitchComputationEnd" ) * 1e6 );
        std::string lStitchRangeFormat = fmt::format( "[{}, {}] \xCE\xBCs", lStitchRange, lStitchRangeWidth );
        mStitchRange->SetValue( lStitchRangeFormat );

        mLocalError->SetValue( "{}", aAcquisitionData->GetPropertyValue<uint32_t>( "LocalErrorType" ) );

        mAverages->SetValue( "{}", aAcquisitionData->GetPropertyValue<uint32_t>( "NumberOfAverages" ) );

        std::string lDecimation =
            fmt::format( "{}", aAcquisitionData->GetPropertyValue<uint32_t>( "NumberOfAddedSamplesDecimation" ) );
        std::string lPhases                = fmt::format( "{}", aAcquisitionData->GetPropertyValue<uint32_t>( "NumberOfPhases" ) );
        std::string lDecimationsOverPhases = fmt::format( "{} / {}", lDecimation, lPhases );
        mDecimationsPhases->SetValue( lDecimationsOverPhases );

        mDeltaT->SetValue( "{:.3f} ns", aAcquisitionData->GetPropertyValue<double>( "DeltaT" ) * 1e9 );

        mTimeToOutput->SetValue( "{:.1} ns", aAcquisitionData->GetPropertyValue<double>( "TimeToOutputConnector" ) * 1e9 );

        mTimeLASToOutput->SetValue( "{:.1f} ns",
                                    aAcquisitionData->GetPropertyValue<double>( "TimeFromLaserToOutputConnector" ) * 1e9 );

        mTTOInternalSampling->SetValue( "{:.1f} ns",
                                        aAcquisitionData->GetPropertyValue<double>( "TimeToOutputConnectorInternalSamples" ) * 1e9 );

        mInternalReflectance->SetValue( "{:.1} dB", aAcquisitionData->GetPropertyValue<double>( "InternalModuleReflection" ) );

        mSamplingDelay->SetValue( "{:.3f} s", aAcquisitionData->GetPropertyValue<double>( "PulseSamplingDelay" ) );

        mRepetitionPeriod->SetValue( "{:.3f} s", aAcquisitionData->GetPropertyValue<double>( "RepetitionPeriod" ) );

        auto lAcquisitionTime = aAcquisitionData->GetPropertyValue<uint32_t>( "NumberOfAverages" ) *
                                aAcquisitionData->GetPropertyValue<uint32_t>( "NumberOfPhases" ) *
                                aAcquisitionData->GetPropertyValue<double>( "RepetitionPeriod" );

        std::string lAcquisitionTimeStr = fmt::format( "{} s", lAcquisitionTime );
        mAcquisitionTime->SetValue( lAcquisitionTimeStr );

        mTzCode->SetValue( "{}", aAcquisitionData->GetPropertyValue<uint32_t>( "TzCode" ) );
        mBandwidth->SetValue( "{:.3f} KHz", aAcquisitionData->GetPropertyValue<double>( "Bandwidth" ) * 0.001f );

        std::string lTypicalAnalogGain =
            fmt::format( "{:.3f} dBo", aAcquisitionData->GetPropertyValue<double>( "TypicalAnalogGain" ) );
        mAnalogGain->SetValue( lTypicalAnalogGain );

        std::string lTypicalApdGain = fmt::format( "{:.3f} dBo", aAcquisitionData->GetPropertyValue<double>( "TypicalApdGain" ) );
        mApdGain->SetValue( lTypicalApdGain );
        mStitchGain->SetValue( "{:.3f} dBo", aAcquisitionData->GetPropertyValue<double>( "NormalizationGainAppliedForStitch" ) );

        std::string lFresnelCorrection = fmt::format( "{:.3f}", aAcquisitionData->GetPropertyValue<double>( "FresnelCorrection" ) );
        mFresnelCorrection->SetValue( lFresnelCorrection );

        mNormalizationFactor->SetValue( "{:.3f}", aAcquisitionData->GetPropertyValue<double>( "NormalizationFactor" ) );
        mHighBandwidthFilter->SetValue( "{:.3f} GHz", aAcquisitionData->GetPropertyValue<double>( "HighBandwidthFilter" ) * 1e-9 );
        mUnfilteredRmsNoise->SetValue( "{:.3f} dB", aAcquisitionData->GetPropertyValue<double>( "UnfilteredRmsNoise" ) );
        mLowBandwidthFilterNoise->SetValue( "{:.3f} dB",
                                            aAcquisitionData->GetPropertyValue<double>( "LowBandwidthFilteredRmsNoise" ) );
        mHighBandwidthFiltedNoise->SetValue( "{:.3f} dB",
                                             aAcquisitionData->GetPropertyValue<double>( "HighBandwidthFilteredRmsNoise" ) );
        mExpectedInjection->SetValue( "{:.3f} dB", aAcquisitionData->GetPropertyValue<double>( "ExpectedInjection" ) );
        mSaturationLevel->SetValue( "{:.3f} dB", aAcquisitionData->GetPropertyValue<double>( "SaturationLevel" ) );
        mFresnelSaturation->SetValue( "{:.3f} dB", aAcquisitionData->GetPropertyValue<double>( "FresnelSaturationLevel" ) );
        mSpeckeNoise->SetValue( "{:.3f}", aAcquisitionData->GetPropertyValue<double>( "SpeckleNoiseFor10nsPulse" ) );
        mSaturationMaskRatio->SetValue( "{}", aAcquisitionData->GetPropertyValue<double>( "SaturationMaskingRatio" ) );
        mSaturationMaskDuration->SetValue( "{}", aAcquisitionData->GetPropertyValue<double>( "SaturationMaskingDuration" ) );
        mApdID->SetValue( "{}", aAcquisitionData->GetPropertyValue<double>( "ApdId" ) );
        mRiseTime->SetValue( "{:.3f} ns", aAcquisitionData->GetPropertyValue<double>( "PulseRiseTime" ) * 1e9 );
        mFallTime->SetValue( "{:.3f} ns", aAcquisitionData->GetPropertyValue<double>( "PulseFallTime" ) * 1e9 );

        auto lEventAnalysisType = DotNetRuntime::GetClassType( "Metrino.Olm.SignalProcessing.EventAnalysis" );
        auto lPositionTolerance = lEventAnalysisType.CallMethod( "ComputeEventPositionTolerance", aSinglePulseTrace->GetInstance() );
        std::string lPositionToleranceStr = fmt::format( "{:.3f} m", *(double *)mono_object_unbox( lPositionTolerance ) );
        mPositionTolerance->SetValue( lPositionToleranceStr );

        const char *lCodes[]     = { "Unknown", "A", "B", "C", "D", "E", "F", "G" };
        const char *lDiameters[] = { "N/A", "N/A", "9", "50", "62.5", "125", "N/A", "N/A" };
        auto        lFiberCode   = aFiberInfo->GetPropertyValue<int32_t>( "FiberCode" );
        auto        lFiberType   = fmt::format( "{} ({} \xCE\xBCm)", lCodes[lFiberCode], lDiameters[lFiberCode] );
        mFiberCode->SetValue( lFiberType );

        mIOR->SetValue( "{:.3f}", aFiberInfo->GetPropertyValue<double>( "Ior" ) );
        mRBS->SetValue( "{:.3f} dB", aFiberInfo->GetPropertyValue<double>( "Rbs" ) );

        mAttenuation->SetValue( "{:.3f} dB/km", aFiberInfo->GetPropertyValue<double>( "TypicalFiberAttenuation" ) * 1000 );

        auto *lSequenceType    = aSinglePulseTrace->GetPropertyValue<MonoString *>( "SequenceType" );
        auto  lSequenceTypeStr = DotNetRuntime::NewString( lSequenceType );
        mSequenceType->SetValue( lSequenceTypeStr );
    }

} // namespace SE::OtdrEditor