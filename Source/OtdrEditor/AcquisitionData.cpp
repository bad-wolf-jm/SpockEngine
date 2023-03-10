#include "AcquisitionData.h"

namespace SE::OtdrEditor
{
    AcquisitionData::AcquisitionData()
        : UIBoxLayout( eBoxLayoutOrientation::VERTICAL )
    {
        const float lItemHeight = 20.0f;

        mSectionTitle = New<UILabel>( "Acquisition data" );
        mSectionTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
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
        mDeltaT = New<UIPropertyValue>( "Delta T:" );
        mSectionPropertyLayout->Add( mDeltaT.get(), lItemHeight, false, true );
        mTimeToOutput = New<UIPropertyValue>( "Time to output:" );
        mSectionPropertyLayout->Add( mTimeToOutput.get(), lItemHeight, false, true );
        mTimeLASToOutput = New<UIPropertyValue>( "Time LAS to output:" );
        mSectionPropertyLayout->Add( mTimeLASToOutput.get(), lItemHeight, false, true );
        mTTOInternalSampling = New<UIPropertyValue>( "TTO internal sam:" );
        mSectionPropertyLayout->Add( mTTOInternalSampling.get(), lItemHeight, false, true );
        mInternalReflectance = New<UIPropertyValue>( "Internal reflectance:" );
        mSectionPropertyLayout->Add( mInternalReflectance.get(), lItemHeight, false, true );
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
        mSectionPropertyLayout->Add(mHighBandwidthFiltedNoise .get(), lItemHeight, false, true );
        mExpectedInjection = New<UIPropertyValue>( "Expected injections:" );
        mSectionPropertyLayout->Add( mExpectedInjection.get(), lItemHeight, false, true );
        mSaturationLevel = New<UIPropertyValue>( "Saturation level:" );
        mSectionPropertyLayout->Add( mSaturationLevel.get(), lItemHeight, false, true );
        mFresnelSaturation = New<UIPropertyValue>( "Fresnel saturationr:" );
        mSectionPropertyLayout->Add( mFresnelSaturation.get(), lItemHeight, false, true );
        mSpeckeNoise = New<UIPropertyValue>( "Speckle noise:" );
        mSectionPropertyLayout->Add( mSpeckeNoise.get(), lItemHeight, false, true );
        mSaturationMaskRatio = New<UIPropertyValue>( "mSaturationMaskRatio:" );
        mSectionPropertyLayout->Add( mSaturationMaskRatio.get(), lItemHeight, false, true );
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
        mSpanOrl = New<UIPropertyValue>( "Apen ORL:" );
        mSectionPropertyLayout->Add( mSpanOrl.get(), lItemHeight, false, true );
        mSequenceType = New<UIPropertyValue>( "Sequence type:" );
        mSectionPropertyLayout->Add( mSequenceType.get(), lItemHeight, false, true );

        Add( mSectionLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
    }

    void AcquisitionData::SetData( Ref<MonoScriptInstance> aRbsData )
    {
        // auto        lTrace      = aRbsData->GetPropertyValue( "Trace", "Metrino.Otdr.SinglePulseTrace" );
        // std::string lPulse      = fmt::format( "{} ns", lTrace->GetPropertyValue<double>( "Pulse" ) * 1e9 );
        // std::string lPulseWidth = fmt::format( "{:.3f} m", lTrace->GetPropertyValue<double>( "PulseWidth" ) );
        // std::string lFormat         = fmt::format( "{} - {}", lPulse, lPulseWidth );
        // mRbsTrace->SetValue( lFormat );

        // std::string lRbsNoise= fmt::format( "{:.2f} dB", aRbsData->GetPropertyValue<double>( "NoiseLevel" ) );
        // mRbsNoise->SetValue( lRbsNoise );

        // std::string lRbsSaturation = fmt::format( "{:.2f} dB", aRbsData->GetPropertyValue<double>( "SaturationLevel" ) );
        // mRbsSaturation->SetValue( lRbsSaturation );

        // auto lLsaData = aRbsData->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );

        // std::string lLsaRange = fmt::format("[{:.3f}, {:.3f}]", lLsaData->GetPropertyValue<double>("StartPosition"),  lLsaData->GetPropertyValue<double>("EndPosition"));
        // mLsaRange->SetValue( lLsaRange );

        // std::string lLsaSlope = fmt::format("{:.3f} dB/km", lLsaData->GetPropertyValue<double>("Slope") * 1000);
        // mLsaSlope->SetValue( lLsaSlope );

        // std::string lLsaOffset = fmt::format("{:.3f} dB", lLsaData->GetPropertyValue<double>("Offset"));
        // mLsaOffset->SetValue( lLsaOffset );

        // std::string lLsaMean = fmt::format("{:.3f} dB", lLsaData->GetPropertyValue<double>("Mean"));
        // mLsaMean->SetValue( lLsaMean );

        // std::string lLsaSlopeError = fmt::format("{:.3f} dB/km", lLsaData->GetPropertyValue<double>("SlopeError"));
        // mLsaSlopeError->SetValue( lLsaSlopeError );

        // std::string lLsaLinear = fmt::format("{}", lLsaData->GetPropertyValue<double>("FitOnLinearData"));
        // mLsaLinear->SetValue( lLsaLinear );

        // std::string lLsaError = fmt::format("{:.3f} dB", lLsaData->GetPropertyValue<double>("RmsError"));
        // mLsaError->SetValue( lLsaError );

        // std::string lLsaCrossings = fmt::format("{}", lLsaData->GetPropertyValue<double>("Crossings"));
        // mLsaCrossings->SetValue( lLsaCrossings );
    }


} // namespace SE::OtdrEditor