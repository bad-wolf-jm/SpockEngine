#include "MeasurementOverview.h"
#include "Mono/MonoRuntime.h"
namespace SE::OtdrEditor
{
    MeasurementOverview::MeasurementOverview()
        : UIBoxLayout( eBoxLayoutOrientation::VERTICAL )
    {

        const float lItemHeight = 20.0f;

        mMeasurementResultTitle = New<UILabel>( "Measurement results" );
        mMeasurementResultTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mMeasurementResultTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mMeasurementResultLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mMeasurementResultPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mMeasurementResultLayout->Add( nullptr, 25.0f, false, false );
        mMeasurementResultLayout->Add( mMeasurementResultPropertyLayout.get(), true, true );
        mOverviewLength = New<UIPropertyValue>( "Overview length:" );
        mMeasurementResultPropertyLayout->Add( mOverviewLength.get(), lItemHeight, false, true );
        mLinkLength = New<UIPropertyValue>( "Link length:" );
        mMeasurementResultPropertyLayout->Add( mLinkLength.get(), lItemHeight, false, true );
        mCompletionStatus = New<UIPropertyValue>( "Completion:" );
        mMeasurementResultPropertyLayout->Add( mCompletionStatus.get(), lItemHeight, false, true );
        Add( mMeasurementResultLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mMeasurementConfigurationTitle = New<UILabel>( "Configuration" );
        mMeasurementConfigurationTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mMeasurementConfigurationTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mMeasurementConfigurationLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mMeasurementConfigurationPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mMeasurementConfigurationLayout->Add( nullptr, 25.0f, false, false );
        mMeasurementConfigurationLayout->Add( mMeasurementConfigurationPropertyLayout.get(), true, true );
        mLaunchFiberLength = New<UIPropertyValue>( "Launch fiber length:" );
        mMeasurementConfigurationPropertyLayout->Add( mLaunchFiberLength.get(), lItemHeight, false, true );
        mConditioner = New<UIPropertyValue>( "Conditioner:" );
        mMeasurementConfigurationPropertyLayout->Add( mConditioner.get(), lItemHeight, false, true );
        mReceiveFiberLength = New<UIPropertyValue>( "Receive fiber length:" );
        mMeasurementConfigurationPropertyLayout->Add( mReceiveFiberLength.get(), lItemHeight, false, true );
        mLoopLength = New<UIPropertyValue>( "Loop length:" );
        mMeasurementConfigurationPropertyLayout->Add( mLoopLength.get(), lItemHeight, false, true );
        mConfigurationFiberCode = New<UIPropertyValue>( "Fiber code:" );
        mMeasurementConfigurationPropertyLayout->Add( mConfigurationFiberCode.get(), lItemHeight, false, true );
        mIOR = New<UIPropertyValue>( "IOR:" );
        mMeasurementConfigurationPropertyLayout->Add( mIOR.get(), lItemHeight, false, true );
        mRBS = New<UIPropertyValue>( "RBS:" );
        mMeasurementConfigurationPropertyLayout->Add( mRBS.get(), lItemHeight, false, true );
        mAttenuation = New<UIPropertyValue>( "Attenuation:" );
        mMeasurementConfigurationPropertyLayout->Add( mAttenuation.get(), lItemHeight, false, true );
        mWavelengths = New<UIPropertyValue>( "Wavelengths:" );
        mMeasurementConfigurationPropertyLayout->Add( mWavelengths.get(), lItemHeight, false, true );
        mDirection = New<UIPropertyValue>( "Direction:" );
        mMeasurementConfigurationPropertyLayout->Add( mDirection.get(), lItemHeight, false, true );
        mLoopbackExtract = New<UIPropertyValue>( "Loopback extract:" );
        mMeasurementConfigurationPropertyLayout->Add( mLoopbackExtract.get(), lItemHeight, false, true );
        mExpectedLinkElements = New<UIPropertyValue>( "Exoected link elements:" );
        mMeasurementConfigurationPropertyLayout->Add( mExpectedLinkElements.get(), lItemHeight, false, true );
        mTopology = New<UIPropertyValue>( "Topology:" );
        mMeasurementConfigurationPropertyLayout->Add( mTopology.get(), lItemHeight, false, true );
        mMeasurementType = New<UIPropertyValue>( "Measurement type:" );
        mMeasurementConfigurationPropertyLayout->Add( mMeasurementType.get(), lItemHeight, false, true );
        mSequenceType = New<UIPropertyValue>( "Sequence type:" );
        mMeasurementConfigurationPropertyLayout->Add( mSequenceType.get(), lItemHeight, false, true );
        mMacrobendThreshold = New<UIPropertyValue>( "Macrobend threshold:" );
        mMeasurementConfigurationPropertyLayout->Add( mMacrobendThreshold.get(), lItemHeight, false, true );
        mComment = New<UIPropertyValue>( "Comment:" );
        mMeasurementConfigurationPropertyLayout->Add( mComment.get(), lItemHeight, false, true );
        mAdvisorMode = New<UIPropertyValue>( "Advisor mode:" );
        mMeasurementConfigurationPropertyLayout->Add( mAdvisorMode.get(), lItemHeight, false, true );
        mDuration = New<UIPropertyValue>( "Duration:" );
        mMeasurementConfigurationPropertyLayout->Add( mDuration.get(), lItemHeight, false, true );
        mDate = New<UIPropertyValue>( "Date:" );
        mMeasurementConfigurationPropertyLayout->Add( mDate.get(), lItemHeight, false, true );
        Add( mMeasurementConfigurationLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mAnalysisResultTitle = New<UILabel>( "Analysis result" );
        mAnalysisResultTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mAnalysisResultTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mAnalysisResultLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mAnalysisResultPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mAnalysisResultLayout->Add( nullptr, 25.0f, false, false );
        mAnalysisResultLayout->Add( mAnalysisResultPropertyLayout.get(), true, true );
        mEOLLinkLoss = New<UIPropertyValue>( "EOL Link Loss:" );
        mAnalysisResultPropertyLayout->Add( mEOLLinkLoss.get(), lItemHeight, false, true );
        mEOLPosition = New<UIPropertyValue>( "EOL Position:" );
        mAnalysisResultPropertyLayout->Add( mEOLPosition.get(), lItemHeight, false, true );
        mProcessingTime = New<UIPropertyValue>( "ProcessingTime:" );
        mAnalysisResultPropertyLayout->Add( mProcessingTime.get(), lItemHeight, false, true );
        Add( mAnalysisResultLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mTestFibersTitle = New<UILabel>( "Test fibers" );
        mTestFibersTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mTestFibersTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mSoftwareVersionsTitle = New<UILabel>( "Software versions" );
        mSoftwareVersionsTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mSoftwareVersionsTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mSoftwareVersionsLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mSoftwareVersionsPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mSoftwareVersionsLayout->Add( nullptr, 25.0f, false, false );
        mSoftwareVersionsLayout->Add( mSoftwareVersionsPropertyLayout.get(), true, true );
        mOtdrInstrumentVersion = New<UIPropertyValue>( "Otdr.Instrument:" );
        mSoftwareVersionsPropertyLayout->Add( mOtdrInstrumentVersion.get(), lItemHeight, false, true );
        mOtdrSignalProcessingVersion = New<UIPropertyValue>( "Otdr.SignalProcessing:" );
        mSoftwareVersionsPropertyLayout->Add( mOtdrSignalProcessingVersion.get(), lItemHeight, false, true );
        mOtdrSimulationVersion = New<UIPropertyValue>( "Otdr.Simulation:" );
        mSoftwareVersionsPropertyLayout->Add( mOtdrSimulationVersion.get(), lItemHeight, false, true );
        mOlmInstrumentVersion = New<UIPropertyValue>( "Olm.Instrument:" );
        mSoftwareVersionsPropertyLayout->Add( mOlmInstrumentVersion.get(), lItemHeight, false, true );
        mOlmSignalProcessingVersion = New<UIPropertyValue>( "Olm.SignalProcessing:" );
        mSoftwareVersionsPropertyLayout->Add( mOlmSignalProcessingVersion.get(), lItemHeight, false, true );
        mIOlmKit = New<UIPropertyValue>( "LLKitiOlm:" );
        mSoftwareVersionsPropertyLayout->Add( mIOlmKit.get(), lItemHeight, false, true );
        Add( mSoftwareVersionsLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mHardwareTitle = New<UILabel>( "Hardware" );
        mHardwareTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mHardwareTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mHardwareLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mHardwarePropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mHardwareLayout->Add( nullptr, 25.0f, false, false );
        mHardwareLayout->Add( mHardwarePropertyLayout.get(), true, true );
        mModelName = New<UIPropertyValue>( "Model name:" );
        mHardwarePropertyLayout->Add( mModelName.get(), lItemHeight, false, true );
        mSerialNumber = New<UIPropertyValue>( "Serial number:" );
        mHardwarePropertyLayout->Add( mSerialNumber.get(), lItemHeight, false, true );
        mOtdrFamily = New<UIPropertyValue>( "OTDR Family:" );
        mHardwarePropertyLayout->Add( mOtdrFamily.get(), lItemHeight, false, true );
        mModelType = New<UIPropertyValue>( "Model type:" );
        mHardwarePropertyLayout->Add( mModelType.get(), lItemHeight, false, true );
        mProductVersion = New<UIPropertyValue>( "Product version:" );
        mHardwarePropertyLayout->Add( mProductVersion.get(), lItemHeight, false, true );
        mPowerLevel = New<UIPropertyValue>( "Power level:" );
        mHardwarePropertyLayout->Add( mPowerLevel.get(), lItemHeight, false, true );
        mFiberCode = New<UIPropertyValue>( "Fiber code:" );
        mHardwarePropertyLayout->Add( mFiberCode.get(), lItemHeight, false, true );
        mOutputPorts = New<UIPropertyValue>( "Output ports:" );
        mHardwarePropertyLayout->Add( mOutputPorts.get(), lItemHeight, false, true );
        mFPGAversion = New<UIPropertyValue>( "FPGA version:" );
        mHardwarePropertyLayout->Add( mFPGAversion.get(), lItemHeight, false, true );
        mPCBVersion = New<UIPropertyValue>( "PCB version:" );
        mHardwarePropertyLayout->Add( mPCBVersion.get(), lItemHeight, false, true );
        mPCBSerialNumber = New<UIPropertyValue>( "PCB serial number:" );
        mHardwarePropertyLayout->Add( mPCBSerialNumber.get(), lItemHeight, false, true );
        mCPUPCBVersion = New<UIPropertyValue>( "CPU PCB version:" );
        mHardwarePropertyLayout->Add( mCPUPCBVersion.get(), lItemHeight, false, true );
        mCPUPCBSerialNumber = New<UIPropertyValue>( "CPU PCB serial number:" );
        mHardwarePropertyLayout->Add( mCPUPCBSerialNumber.get(), lItemHeight, false, true );
        mPowerMeterVersion = New<UIPropertyValue>( "Power meter version:" );
        mHardwarePropertyLayout->Add( mPowerMeterVersion.get(), lItemHeight, false, true );
        mPowerMeterSerialNumber = New<UIPropertyValue>( "Power meter serial number:" );
        mHardwarePropertyLayout->Add( mPowerMeterSerialNumber.get(), lItemHeight, false, true );
        mManufacturingDate = New<UIPropertyValue>( "Manufacturing date:" );
        mHardwarePropertyLayout->Add( mManufacturingDate.get(), lItemHeight, false, true );
        mLastCalibrationDate = New<UIPropertyValue>( "Calibration Date:" );
        mHardwarePropertyLayout->Add( mLastCalibrationDate.get(), lItemHeight, false, true );
        mUserLastCalibrationDate = New<UIPropertyValue>( "User Calibration Date:" );
        mHardwarePropertyLayout->Add( mUserLastCalibrationDate.get(), lItemHeight, false, true );
        Add( mHardwareLayout.get(), true, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
    }

    void MeasurementOverview::SetData( Ref<MonoScriptInstance> aMeasurementOverview )
    {
        auto lMeasurementOverview = aMeasurementOverview->GetPropertyValue( "Measurement", "Metrino.Olm.OlmMeasurement" );

        // Overview
        std::string lMeasurementLength =
            fmt::format( "{:.4f} km", lMeasurementOverview->GetPropertyValue<double>( "OverviewLength" ) * 0.001f );
        mOverviewLength->SetValue( lMeasurementLength );

        std::string lLinkLength = fmt::format( "{:.4f} km", lMeasurementOverview->GetPropertyValue<double>( "LinkLength" ) * 0.001f );
        mLinkLength->SetValue( lLinkLength );

        // Configuration
        std::string lLaunchFiberLength =
            fmt::format( "{:.4f} km", lMeasurementOverview->GetPropertyValue<double>( "LaunchFiberLength" ) * 0.001f );
        mLaunchFiberLength->SetValue( lLaunchFiberLength );

        std::string lLaunchConditioner =
            fmt::format( "{} ENUM", lMeasurementOverview->GetPropertyValue<int>( "EfLaunchConditioner" ) );
        mConditioner->SetValue( lLaunchConditioner );

        std::string lReceiveFiberLength =
            fmt::format( "{:.4f} km", lMeasurementOverview->GetPropertyValue<double>( "ReceiveFiberLength" ) * 0.001f );
        mReceiveFiberLength->SetValue( lReceiveFiberLength );

        std::string lLoopLength = fmt::format( "{:.4f} km", lMeasurementOverview->GetPropertyValue<double>( "LoopLength" ) * 0.001f );
        mLoopLength->SetValue( lLoopLength );

        auto lOtdrParameters = aMeasurementOverview->GetPropertyValue( "OtdrParameters", "Metrino.Olm.OtdrParameters" );
        auto lFiberInfo      = aMeasurementOverview->GetPropertyValue( "FiberInfo", "Metrino.Otdr.PhysicalFiberCharacteristics" );
        auto lWavelength     = lOtdrParameters->GetPropertyValue<double>( "Wavelength" );

        const char *lCodes[]     = { "Unknown", "A", "B", "C", "D", "E", "F", "G" };
        const char *lDiameters[] = { "N/A", "N/A", "9", "50", "62.5", "125", "N/A", "N/A" };
        auto        lFiberCode   = lFiberInfo->GetPropertyValue<int32_t>( "FiberCode" );
        auto        lFiberType   = fmt::format( "{} ({} \xCE\xBCm)", lCodes[lFiberCode], lDiameters[lFiberCode] );
        mConfigurationFiberCode->SetValue( lFiberType );

        std::string lIOR = fmt::format( "{:.3f}", lFiberInfo->GetPropertyValue<double>( "Ior" ) );
        mIOR->SetValue( lIOR );

        std::string lRbs = fmt::format( "{:.3f} dB", lFiberInfo->GetPropertyValue<double>( "Rbs" ) );
        mRBS->SetValue( lRbs );

        std::string lTypicalFiberAttenuation =
            fmt::format( "{:.3f} dB/km", lFiberInfo->GetPropertyValue<double>( "TypicalFiberAttenuation" ) * 1000 );
        mAttenuation->SetValue( lTypicalFiberAttenuation );

        std::string lDirection = fmt::format( "{}", " A\xE2\x86\x92 B (Not Implemented)" );
        mDirection->SetValue( lDirection );

        std::string lLoopbackExtract = fmt::format( "{}", lMeasurementOverview->GetPropertyValue<bool>( "LoopbackExtract" ) );
        mLoopbackExtract->SetValue( lLoopbackExtract );

        // Analysis results

        // Software versions

        // Hardware versions
        auto lUnitInfo = lMeasurementOverview->GetPropertyValue( "UnitA", "Metrino.Kernos.Optical.UnitInformation" );

        // auto lModelName    = lUnitInfo->GetPropertyValue<MonoString *>( "ModelName" );
        // auto lModelNameStr = MonoRuntime::NewString( lModelName );
        mModelName->SetValue( "!!!TODO!!!" );

        // auto lSeriualNumber    = lUnitInfo->GetPropertyValue<MonoString *>( "SerialNumber" );
        // auto lSeriualNumberStr = MonoRuntime::NewString( lSeriualNumber );
        mSerialNumber->SetValue( "!!!TODO!!!" );

        mOtdrFamily->SetValue( "!!!TODO!!!" );
        mModelType->SetValue( "!!!TODO!!!" );
        mProductVersion->SetValue( "!!!TODO!!!" );
        mPowerLevel->SetValue( "!!!TODO!!!" );
        mFiberCode->SetValue( "!!!TODO!!!" );
        mOutputPorts->SetValue( "!!!TODO!!!" );
        mFPGAversion->SetValue( "!!!TODO!!!" );
        mPCBVersion->SetValue( "!!!TODO!!!" );
        mPCBSerialNumber->SetValue( "!!!TODO!!!" );
        mCPUPCBVersion->SetValue( "!!!TODO!!!" );
        mCPUPCBSerialNumber->SetValue( "!!!TODO!!!" );
        mPowerMeterVersion->SetValue( "!!!TODO!!!" );
        mPowerMeterSerialNumber->SetValue( "!!!TODO!!!" );
        mManufacturingDate->SetValue( "!!!TODO!!!" );
        mLastCalibrationDate->SetValue( "!!!TODO!!!" );
        mUserLastCalibrationDate->SetValue( "!!!TODO!!!" );
    }
} // namespace SE::OtdrEditor