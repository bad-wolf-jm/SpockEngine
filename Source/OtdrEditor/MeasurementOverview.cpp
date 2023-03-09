#include "MeasurementOverview.h"

namespace SE::OtdrEditor
{
    PropertyValue::PropertyValue( std::string aName )
        : UIBoxLayout( eBoxLayoutOrientation::HORIZONTAL )
    {
        mName  = New<UILabel>( aName );
        mValue = New<UILabel>();

        Add( mName.get(), true, false, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mValue.get(), true, false, eHorizontalAlignment::RIGHT, eVerticalAlignment::CENTER );
    }

    void PropertyValue::SetValue( std::string aValue ) { mValue->SetText( aValue ); }

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
        mOverviewLength = New<PropertyValue>( "Overview length:" );
        mMeasurementResultPropertyLayout->Add( mOverviewLength.get(), 20.0f, false, true );
        mLinkLength = New<PropertyValue>( "Link length:" );
        mMeasurementResultPropertyLayout->Add( mLinkLength.get(), 20.0f, false, true );
        mCompletionStatus = New<PropertyValue>( "Completion:" );
        mMeasurementResultPropertyLayout->Add( mCompletionStatus.get(), 20.0f, false, true );
        Add( mMeasurementResultLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mMeasurementConfigurationTitle = New<UILabel>( "Configuration" );
        mMeasurementConfigurationTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mMeasurementConfigurationTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mMeasurementConfigurationLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mMeasurementConfigurationPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mMeasurementConfigurationLayout->Add( nullptr, 25.0f, false, false );
        mMeasurementConfigurationLayout->Add( mMeasurementConfigurationPropertyLayout.get(), true, true );

        mAnalysisResultTitle = New<UILabel>( "Analysis result" );
        mAnalysisResultTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mAnalysisResultTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mAnalysisResultLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mAnalysisResultPropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mAnalysisResultLayout->Add( nullptr, 25.0f, false, false );
        mAnalysisResultLayout->Add( mAnalysisResultPropertyLayout.get(), true, true );
        mEOLLinkLoss = New<PropertyValue>( "EOL Link Loss:" );
        mAnalysisResultPropertyLayout->Add( mEOLLinkLoss.get(), 20.0f, false, true );
        mEOLPosition = New<PropertyValue>( "EOL Position:" );
        mAnalysisResultPropertyLayout->Add( mEOLPosition.get(), 20.0f, false, true );
        mProcessingTime = New<PropertyValue>( "ProcessingTime:" );
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
        mOtdrInstrumentVersion = New<PropertyValue>( "Otdr.Instrument:" );
        mSoftwareVersionsPropertyLayout->Add( mOtdrInstrumentVersion.get(), 20.0f, false, true );
        mOtdrSignalProcessingVersion = New<PropertyValue>( "Otdr.SignalProcessing:" );
        mSoftwareVersionsPropertyLayout->Add( mOtdrSignalProcessingVersion.get(), 20.0f, false, true );
        mOtdrSimulationVersion = New<PropertyValue>( "Otdr.Simulation:" );
        mSoftwareVersionsPropertyLayout->Add( mOtdrSimulationVersion.get(), 20.0f, false, true );
        mOlmInstrumentVersion = New<PropertyValue>( "Olm.Instrument:" );
        mSoftwareVersionsPropertyLayout->Add( mOlmInstrumentVersion.get(), 20.0f, false, true );
        mOlmSignalProcessingVersion = New<PropertyValue>( "Olm.SignalProcessing:" );
        mSoftwareVersionsPropertyLayout->Add( mOlmSignalProcessingVersion.get(), 20.0f, false, true );
        mIOlmKit = New<PropertyValue>( "LLKitiOlm:" );
        mSoftwareVersionsPropertyLayout->Add( mIOlmKit.get(), 20.0f, false, true );
        Add( mSoftwareVersionsLayout.get(), false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );

        mHardwareTitle = New<UILabel>( "Hardware" );
        mHardwareTitle->SetAlignment( eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        Add( mHardwareTitle.get(), 30.0f, false, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
        mHardwareLayout         = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mHardwarePropertyLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mHardwareLayout->Add( nullptr, 25.0f, false, false );
        mHardwareLayout->Add( mHardwarePropertyLayout.get(), true, true );
        mModelName = New<PropertyValue>( "Model name:" );
        mHardwarePropertyLayout->Add( mModelName.get(), 20.0f, false, true );
        mSerialNumber = New<PropertyValue>( "Serial number:" );
        mHardwarePropertyLayout->Add( mSerialNumber.get(), 20.0f, false, true );
        mOtdrFamily = New<PropertyValue>( "OTDR Family:" );
        mHardwarePropertyLayout->Add( mOtdrFamily.get(), 20.0f, false, true );
        mModelType = New<PropertyValue>( "Model type:" );
        mHardwarePropertyLayout->Add( mModelType.get(), 20.0f, false, true );
        mProductVersion = New<PropertyValue>( "Product version:" );
        mHardwarePropertyLayout->Add( mProductVersion.get(), 20.0f, false, true );
        mPowerLevel = New<PropertyValue>( "Power level:" );
        mHardwarePropertyLayout->Add( mPowerLevel.get(), 20.0f, false, true );
        mFiberCode = New<PropertyValue>( "Fiber code:" );
        mHardwarePropertyLayout->Add( mFiberCode.get(), 20.0f, false, true );
        mOutputPorts = New<PropertyValue>( "Output ports:" );
        mHardwarePropertyLayout->Add( mOutputPorts.get(), 20.0f, false, true );
        mFPGAversion = New<PropertyValue>( "FPGA version:" );
        mHardwarePropertyLayout->Add( mFPGAversion.get(), 20.0f, false, true );
        mPCBVersion = New<PropertyValue>( "PCB version:" );
        mHardwarePropertyLayout->Add( mPCBVersion.get(), 20.0f, false, true );
        mPCBSerialNumber = New<PropertyValue>( "PCB serial number:" );
        mHardwarePropertyLayout->Add( mPCBSerialNumber.get(), 20.0f, false, true );
        mCPUPCBVersion = New<PropertyValue>( "CPUPCB version:" );
        mHardwarePropertyLayout->Add( mCPUPCBVersion.get(), 20.0f, false, true );
        mCPUPCBSerialNumber = New<PropertyValue>( "CPUPCB serial number:" );
        mHardwarePropertyLayout->Add( mCPUPCBSerialNumber.get(), 20.0f, false, true );
        mPowerMeterVersion = New<PropertyValue>( "PM version:" );
        mHardwarePropertyLayout->Add( mPowerMeterVersion.get(), 20.0f, false, true );
        mPowerMeterSerialNumber = New<PropertyValue>( "PM serial number:" );
        mHardwarePropertyLayout->Add( mPowerMeterSerialNumber.get(), 20.0f, false, true );
        mManufacturingDate = New<PropertyValue>( "Manufacturing date:" );
        mHardwarePropertyLayout->Add( mManufacturingDate.get(), 20.0f, false, true );
        mLastCalibrationDate = New<PropertyValue>( "Calibration Date:" );
        mHardwarePropertyLayout->Add( mLastCalibrationDate.get(), 20.0f, false, true );
        mUserLastCalibrationDate = New<PropertyValue>( "User Calibration Date:" );
        mHardwarePropertyLayout->Add( mUserLastCalibrationDate.get(), 20.0f, false, true );
        Add( mHardwareLayout.get(), true, true, eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER );
    }

} // namespace SE::OtdrEditor