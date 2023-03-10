#include "MeasurementOverview.h"

namespace SE::OtdrEditor
{
    UIPropertyValue::UIPropertyValue( std::string aName )
        : UIBoxLayout( eBoxLayoutOrientation::HORIZONTAL )
    {
        mName  = New<UILabel>( aName );
        mValue = New<UILabel>( "N/A" );

        Add( mName.get(), true, false, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mValue.get(), true, false, eHorizontalAlignment::RIGHT, eVerticalAlignment::CENTER );
    }

    void UIPropertyValue::SetValue( std::string aValue ) { mValue->SetText( aValue ); }

    MeasurementOverview::MeasurementOverview()
        : UIBoxLayout( eBoxLayoutOrientation::VERTICAL )
    {
        mMeasurementResultTitle = New<UILabel>( "Measurement results" );
        mMeasurementResultTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mMeasurementResultTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mMeasurementResultLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mMeasurementResultPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mMeasurementResultLayout->Add( nullptr, 25.0f, false, false );
        mMeasurementResultLayout->Add( mMeasurementResultPropertyLayout.get(), true, true );
        mOverviewLength = New<UIPropertyValue>( "Overview length:" );
        mMeasurementResultPropertyLayout->Add( mOverviewLength.get(), 20.0f, false, true );
        mLinkLength = New<UIPropertyValue>( "Link length:" );
        mMeasurementResultPropertyLayout->Add( mLinkLength.get(), 20.0f, false, true );
        mCompletionStatus = New<UIPropertyValue>( "Completion:" );
        mMeasurementResultPropertyLayout->Add( mCompletionStatus.get(), 20.0f, false, true );
        Add( mMeasurementResultLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mMeasurementConfigurationTitle = New<UILabel>( "Configuration" );
        mMeasurementConfigurationTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mMeasurementConfigurationTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mMeasurementConfigurationLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mMeasurementConfigurationPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mMeasurementConfigurationLayout->Add( nullptr, 25.0f, false, false );
        mMeasurementConfigurationLayout->Add( mMeasurementConfigurationPropertyLayout.get(), true, true );
        mLaunchFiberLength = New<UIPropertyValue>( "Launch fiber length:" );
        mMeasurementConfigurationPropertyLayout->Add( mLaunchFiberLength.get(), 20.0f, false, true );
        mConditioner = New<UIPropertyValue>( "Conditioner:" );
        mMeasurementConfigurationPropertyLayout->Add( mConditioner.get(), 20.0f, false, true );
        mReceiveFiberLength = New<UIPropertyValue>( "Receive fiber length:" );
        mMeasurementConfigurationPropertyLayout->Add( mReceiveFiberLength.get(), 20.0f, false, true );
        mLoopLength = New<UIPropertyValue>( "Loop length:" );
        mMeasurementConfigurationPropertyLayout->Add( mLoopLength.get(), 20.0f, false, true );
        mConfigurationFiberCode = New<UIPropertyValue>( "Fiber code:" );
        mMeasurementConfigurationPropertyLayout->Add( mConfigurationFiberCode.get(), 20.0f, false, true );
        mIOR = New<UIPropertyValue>( "IOR:" );
        mMeasurementConfigurationPropertyLayout->Add( mIOR.get(), 20.0f, false, true );
        mRBS = New<UIPropertyValue>( "RBS:" );
        mMeasurementConfigurationPropertyLayout->Add( mRBS.get(), 20.0f, false, true );
        mAttenuation = New<UIPropertyValue>( "Attenuation:" );
        mMeasurementConfigurationPropertyLayout->Add( mAttenuation.get(), 20.0f, false, true );
        mWavelengths = New<UIPropertyValue>( "Wavelengths:" );
        mMeasurementConfigurationPropertyLayout->Add( mWavelengths.get(), 20.0f, false, true );
        mDirection = New<UIPropertyValue>( "Direction:" );
        mMeasurementConfigurationPropertyLayout->Add( mDirection.get(), 20.0f, false, true );
        mLoopbackExtract = New<UIPropertyValue>( "Loopback extract:" );
        mMeasurementConfigurationPropertyLayout->Add( mLoopbackExtract.get(), 20.0f, false, true );
        mExpectedLinkElements = New<UIPropertyValue>( "Exoected link elements:" );
        mMeasurementConfigurationPropertyLayout->Add( mExpectedLinkElements.get(), 20.0f, false, true );
        mTopology = New<UIPropertyValue>( "Topology:" );
        mMeasurementConfigurationPropertyLayout->Add( mTopology.get(), 20.0f, false, true );
        mMeasurementType = New<UIPropertyValue>( "Measurement type:" );
        mMeasurementConfigurationPropertyLayout->Add( mMeasurementType.get(), 20.0f, false, true );
        mSequenceType = New<UIPropertyValue>( "Sequence type:" );
        mMeasurementConfigurationPropertyLayout->Add( mSequenceType.get(), 20.0f, false, true );
        mMacrobendThreshold = New<UIPropertyValue>( "Macrobend threshold:" );
        mMeasurementConfigurationPropertyLayout->Add( mMacrobendThreshold.get(), 20.0f, false, true );
        mComment = New<UIPropertyValue>( "Comment:" );
        mMeasurementConfigurationPropertyLayout->Add( mComment.get(), 20.0f, false, true );
        mAdvisorMode = New<UIPropertyValue>( "Advisor mode:" );
        mMeasurementConfigurationPropertyLayout->Add( mAdvisorMode.get(), 20.0f, false, true );
        mDuration = New<UIPropertyValue>( "Duration:" );
        mMeasurementConfigurationPropertyLayout->Add( mDuration.get(), 20.0f, false, true );
        mDate = New<UIPropertyValue>( "Date:" );
        mMeasurementConfigurationPropertyLayout->Add( mDate.get(), 20.0f, false, true );
        Add( mMeasurementConfigurationLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mAnalysisResultTitle = New<UILabel>( "Analysis result" );
        mAnalysisResultTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mAnalysisResultTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mAnalysisResultLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mAnalysisResultPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mAnalysisResultLayout->Add( nullptr, 25.0f, false, false );
        mAnalysisResultLayout->Add( mAnalysisResultPropertyLayout.get(), true, true );
        mEOLLinkLoss = New<UIPropertyValue>( "EOL Link Loss:" );
        mAnalysisResultPropertyLayout->Add( mEOLLinkLoss.get(), 20.0f, false, true );
        mEOLPosition = New<UIPropertyValue>( "EOL Position:" );
        mAnalysisResultPropertyLayout->Add( mEOLPosition.get(), 20.0f, false, true );
        mProcessingTime = New<UIPropertyValue>( "ProcessingTime:" );
        mAnalysisResultPropertyLayout->Add( mProcessingTime.get(), 20.0f, false, true );
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
        mSoftwareVersionsPropertyLayout->Add( mOtdrInstrumentVersion.get(), 20.0f, false, true );
        mOtdrSignalProcessingVersion = New<UIPropertyValue>( "Otdr.SignalProcessing:" );
        mSoftwareVersionsPropertyLayout->Add( mOtdrSignalProcessingVersion.get(), 20.0f, false, true );
        mOtdrSimulationVersion = New<UIPropertyValue>( "Otdr.Simulation:" );
        mSoftwareVersionsPropertyLayout->Add( mOtdrSimulationVersion.get(), 20.0f, false, true );
        mOlmInstrumentVersion = New<UIPropertyValue>( "Olm.Instrument:" );
        mSoftwareVersionsPropertyLayout->Add( mOlmInstrumentVersion.get(), 20.0f, false, true );
        mOlmSignalProcessingVersion = New<UIPropertyValue>( "Olm.SignalProcessing:" );
        mSoftwareVersionsPropertyLayout->Add( mOlmSignalProcessingVersion.get(), 20.0f, false, true );
        mIOlmKit = New<UIPropertyValue>( "LLKitiOlm:" );
        mSoftwareVersionsPropertyLayout->Add( mIOlmKit.get(), 20.0f, false, true );
        Add( mSoftwareVersionsLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mHardwareTitle = New<UILabel>( "Hardware" );
        mHardwareTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mHardwareTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mHardwareLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mHardwarePropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mHardwareLayout->Add( nullptr, 25.0f, false, false );
        mHardwareLayout->Add( mHardwarePropertyLayout.get(), true, true );
        mModelName = New<UIPropertyValue>( "Model name:" );
        mHardwarePropertyLayout->Add( mModelName.get(), 20.0f, false, true );
        mSerialNumber = New<UIPropertyValue>( "Serial number:" );
        mHardwarePropertyLayout->Add( mSerialNumber.get(), 20.0f, false, true );
        mOtdrFamily = New<UIPropertyValue>( "OTDR Family:" );
        mHardwarePropertyLayout->Add( mOtdrFamily.get(), 20.0f, false, true );
        mModelType = New<UIPropertyValue>( "Model type:" );
        mHardwarePropertyLayout->Add( mModelType.get(), 20.0f, false, true );
        mProductVersion = New<UIPropertyValue>( "Product version:" );
        mHardwarePropertyLayout->Add( mProductVersion.get(), 20.0f, false, true );
        mPowerLevel = New<UIPropertyValue>( "Power level:" );
        mHardwarePropertyLayout->Add( mPowerLevel.get(), 20.0f, false, true );
        mFiberCode = New<UIPropertyValue>( "Fiber code:" );
        mHardwarePropertyLayout->Add( mFiberCode.get(), 20.0f, false, true );
        mOutputPorts = New<UIPropertyValue>( "Output ports:" );
        mHardwarePropertyLayout->Add( mOutputPorts.get(), 20.0f, false, true );
        mFPGAversion = New<UIPropertyValue>( "FPGA version:" );
        mHardwarePropertyLayout->Add( mFPGAversion.get(), 20.0f, false, true );
        mPCBVersion = New<UIPropertyValue>( "PCB version:" );
        mHardwarePropertyLayout->Add( mPCBVersion.get(), 20.0f, false, true );
        mPCBSerialNumber = New<UIPropertyValue>( "PCB serial number:" );
        mHardwarePropertyLayout->Add( mPCBSerialNumber.get(), 20.0f, false, true );
        mCPUPCBVersion = New<UIPropertyValue>( "CPU PCB version:" );
        mHardwarePropertyLayout->Add( mCPUPCBVersion.get(), 20.0f, false, true );
        mCPUPCBSerialNumber = New<UIPropertyValue>( "CPU PCB serial number:" );
        mHardwarePropertyLayout->Add( mCPUPCBSerialNumber.get(), 20.0f, false, true );
        mPowerMeterVersion = New<UIPropertyValue>( "Power meter version:" );
        mHardwarePropertyLayout->Add( mPowerMeterVersion.get(), 20.0f, false, true );
        mPowerMeterSerialNumber = New<UIPropertyValue>( "Power meter serial number:" );
        mHardwarePropertyLayout->Add( mPowerMeterSerialNumber.get(), 20.0f, false, true );
        mManufacturingDate = New<UIPropertyValue>( "Manufacturing date:" );
        mHardwarePropertyLayout->Add( mManufacturingDate.get(), 20.0f, false, true );
        mLastCalibrationDate = New<UIPropertyValue>( "Calibration Date:" );
        mHardwarePropertyLayout->Add( mLastCalibrationDate.get(), 20.0f, false, true );
        mUserLastCalibrationDate = New<UIPropertyValue>( "User Calibration Date:" );
        mHardwarePropertyLayout->Add( mUserLastCalibrationDate.get(), 20.0f, false, true );
        Add( mHardwareLayout.get(), true, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
    }

    void MeasurementOverview::SetData( Ref<MonoScriptInstance> aMeasurementOverview )
    {
        // Overview
        std::string lMeasurementLength =
            fmt::format( "{:.4f} km", aMeasurementOverview->GetPropertyValue<double>( "OverviewLength" ) * 0.001f );
        mOverviewLength->SetValue( lMeasurementLength );

        std::string lLinkLength = fmt::format( "{:.4f} km", aMeasurementOverview->GetPropertyValue<double>( "LinkLength" ) * 0.001f );
        mLinkLength->SetValue( lLinkLength );

        // Configuration
        std::string lLaunchFiberLength =
            fmt::format( "{:.4f} km", aMeasurementOverview->GetPropertyValue<double>( "LaunchFiberLength" ) * 0.001f );
        mLaunchFiberLength->SetValue( lLaunchFiberLength );

        std::string lReceiveFiberLength =
            fmt::format( "{:.4f} km", aMeasurementOverview->GetPropertyValue<double>( "ReceiveFiberLength" ) * 0.001f );
        mReceiveFiberLength->SetValue( lReceiveFiberLength );

        std::string lLoopLength = fmt::format( "{:.4f} km", aMeasurementOverview->GetPropertyValue<double>( "LoopLength" ) * 0.001f );
        mLoopLength->SetValue( lLoopLength );

        // Analysis results

        // Software versions

        // Hardware versions
        
    }
} // namespace SE::OtdrEditor