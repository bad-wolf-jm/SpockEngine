#include "IOlmDiffDocument.h"

#include "DotNet/Runtime.h"

namespace SE::OtdrEditor
{

    struct sDotNetLinkElement
    {
        int       mRowIndex;
        int       mLinkIndex;
        int       mSubLinkIndex;
        int       mEventIndex;
        int       mSubEventIndex;
        bool      mIsSubElement;
        int       mDiagnosicCount;
        ePassFail mLossPassFail;
        ePassFail mReflectancePassFail;
        void     *mLinkElement;
        void     *mPhysicalEvent;
        void     *mPeakTrace;
        void     *mDetectionTrace;
        void     *mAttributes;
        void     *mAcquisitionData;
        void     *mFiberInfo;
    };

    UIIolmDiffDocument::UIIolmDiffDocument( fs::path aPath, bool aReanalyse )
    {
        mOpen  = true;
        mDirty = false;
        mName  = aPath.filename().string();

        mTracePlot         = New<UILinkElementTracePlot>();
        mLinkElementTable0 = New<UILinkElementTable>();
        mLinkElementTable1 = New<UILinkElementTable>();

        mTopLayout = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mTopLayout->SetItemSpacing( 15.0f );

        mLaunchFiberLength = New<UIPropertyValue>( "Launch Fiber Length", eBoxLayoutOrientation::VERTICAL );
        mLaunchFiberLength->SetBackgroundColor( math::vec4{ 0.025f, 0.025f, 0.025f, 1.0f } );
        mLaunchFiberLength->SetValueFont( FontFamilyFlags::DISPLAY );
        mLaunchFiberLength->SetPadding(10);
        mTopLayout->Add( mLaunchFiberLength.get(), true, true );

        mReceiveFiberLength = New<UIPropertyValue>( "Receive Fiber Length", eBoxLayoutOrientation::VERTICAL );
        mReceiveFiberLength->SetBackgroundColor( math::vec4{ 0.025f, 0.025f, 0.025f, 1.0f } );
        mReceiveFiberLength->SetValueFont( FontFamilyFlags::DISPLAY );
        mReceiveFiberLength->SetPadding(10);
        mTopLayout->Add( mReceiveFiberLength.get(), true, true );

        mLinkLength = New<UIPropertyValue>( "Link Length", eBoxLayoutOrientation::VERTICAL );
        mLinkLength->SetBackgroundColor( math::vec4{ 0.025f, 0.025f, 0.025f, 1.0f } );
        mLinkLength->SetValueFont( FontFamilyFlags::DISPLAY );
        mLinkLength->SetPadding(10);
        mTopLayout->Add( mLinkLength.get(), true, true );

        mFiberCode = New<UIPropertyValue>( "Fiber Code", eBoxLayoutOrientation::VERTICAL );
        mFiberCode->SetBackgroundColor( math::vec4{ 0.025f, 0.025f, 0.025f, 1.0f } );
        mFiberCode->SetValueFont( FontFamilyFlags::DISPLAY );
        mFiberCode->SetPadding(10);
        mTopLayout->Add( mFiberCode.get(), true, true );

        mEventLayout = New<UIBoxLayout>( eBoxLayoutOrientation::HORIZONTAL );
        mEventLayout->Add( mLinkElementTable0.get(), true, true );
        mEventLayout->Add( mLinkElementTable1.get(), true, true );

        mMainLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mMainLayout->SetItemSpacing( 15.0f );
        mMainLayout->Add( mTopLayout.get(), 60.0f, true, true );
        mMainLayout->Add( mEventLayout.get(), true, true );
        mMainLayout->Add( mTracePlot.get(), true, true );

        SetContent( mMainLayout.get() );

        mLinkElementTable0->OnElementClicked(
            [&]( sLinkElement const &aElement )
            {
                auto lWavelength = aElement.mPhysicalEvent->GetPropertyValue<double>( "Wavelength" ) * 1e9;
                mTracePlot->Clear();
                mTracePlot->SetEventData( aElement, true, true, true );
            } );

        static auto &lFileLoader = DotNetRuntime::GetClassType( "Metrino.Interop.FileLoader" );
        static auto &lFileClass  = DotNetRuntime::GetClassType( "Metrino.Interop.OlmFile" );

        MonoString *lFilePath   = DotNetRuntime::NewString( aPath.string() );
        MonoObject *lDataObject = lFileLoader.CallMethod( "LoadOlmData", lFilePath );

        mDataInstance = New<DotNetInstance>( &lFileClass, lFileClass.Class(), lDataObject );

        MonoObject               *lTraceData       = mDataInstance->CallMethod( "GetAllTraces" );
        std::vector<MonoObject *> lTraceDataVector = DotNetRuntime::AsVector<MonoObject *>( lTraceData );
        if( lTraceDataVector.size() != 0 )
        {
            static auto &lTraceDataStructure        = DotNetRuntime::GetClassType( "Metrino.Interop.TracePlotData" );
            static auto &lSinglePulseTraceClassType = DotNetRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );
            static auto &lAcquisitionDataClassType  = DotNetRuntime::GetClassType( "Metrino.Otdr.AcquisitionData" );
            static auto &lFiberInfoClassType        = DotNetRuntime::GetClassType( "Metrino.Otdr.PhysicalFiberCharacteristics" );

            auto &lTraceDataInstance = DotNetInstance( &lTraceDataStructure, lTraceDataStructure.Class(), lTraceDataVector[0] );

            auto lSinglePulseTrace = lTraceDataInstance.GetFieldValue<MonoObject *>( "mTrace" );
            auto lSinglePulseTraceInstance =
                New<DotNetInstance>( &lSinglePulseTraceClassType, lSinglePulseTraceClassType.Class(), lSinglePulseTrace );

            auto lAcquisitionData = lTraceDataInstance.GetFieldValue<MonoObject *>( "mAcquisitionData" );
            auto lAcquisitionDataInstance =
                New<DotNetInstance>( &lAcquisitionDataClassType, lAcquisitionDataClassType.Class(), lAcquisitionData );

            auto lFiberInfo = lTraceDataInstance.GetPropertyValue( "FiberInfo", "Metrino.Otdr.PhysicalFiberCharacteristics" );
        }

        auto lMeasurementOverview = mDataInstance->GetPropertyValue( "Measurement", "Metrino.Olm.OlmMeasurement" );

        std::string lLinkLength = fmt::format( "{:.4f} km", lMeasurementOverview->GetPropertyValue<double>( "LinkLength" ) * 0.001f );
        mLinkLength->SetValue( lLinkLength );

        std::string lLaunchFiberLength =
            fmt::format( "{:.4f} km", lMeasurementOverview->GetPropertyValue<double>( "LaunchFiberLength" ) * 0.001f );
        mLaunchFiberLength->SetValue( lLaunchFiberLength );
        std::string lReceiveFiberLength =
            fmt::format( "{:.4f} km", lMeasurementOverview->GetPropertyValue<double>( "ReceiveFiberLength" ) * 0.001f );
        mReceiveFiberLength->SetValue( lReceiveFiberLength );

        auto lFiberInfo = mDataInstance->GetPropertyValue( "FiberInfo", "Metrino.Otdr.PhysicalFiberCharacteristics" );

        const char *lCodes[]     = { "Unknown", "A", "B", "C", "D", "E", "F", "G" };
        const char *lDiameters[] = { "N/A", "N/A", "9", "50", "62.5", "125", "N/A", "N/A" };
        auto        lFiberCode   = lFiberInfo->GetPropertyValue<int32_t>( "FiberCode" );
        auto        lFiberType   = fmt::format( "{} ({} \xCE\xBCm)", lCodes[lFiberCode], lDiameters[lFiberCode] );
        mFiberCode->SetValue( lFiberType );

        {
            auto        lReanalyze       = false;
            MonoObject *lLinkElementData = mDataInstance->CallMethod( "GetLinkElements", &lReanalyze );

            auto lLinkElementVector = DotNetRuntime::AsVector<sDotNetLinkElement>( lLinkElementData );

            mLinkElementVector0 = std::vector<sLinkElement>();

            for( auto const &x : lLinkElementVector )
            {
                auto &lElement = mLinkElementVector0.emplace_back();

                lElement.mRowIndex            = x.mRowIndex;
                lElement.mLinkIndex           = x.mLinkIndex;
                lElement.mSubLinkIndex        = x.mSubLinkIndex;
                lElement.mEventIndex          = x.mEventIndex;
                lElement.mSubEventIndex       = x.mSubEventIndex;
                lElement.mIsSubElement        = x.mIsSubElement;
                lElement.mDiagnosicCount      = x.mDiagnosicCount;
                lElement.mLossPassFail        = x.mLossPassFail;
                lElement.mReflectancePassFail = x.mReflectancePassFail;

                static auto &lBaseLinkElementClass  = DotNetRuntime::GetClassType( "Metrino.Olm.BaseLinkElement" );
                static auto &lOlmPhysicalEventClass = DotNetRuntime::GetClassType( "Metrino.Olm.OlmPhysicalEvent" );
                static auto &lOlmAttributeClass =
                    DotNetRuntime::GetClassType( "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute" );
                static auto &lAcquisitionDataClassType = DotNetRuntime::GetClassType( "Metrino.Otdr.AcquisitionData" );
                static auto &lSinglePulseTraceClass    = DotNetRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );
                static auto &lFiberInfoClassType       = DotNetRuntime::GetClassType( "Metrino.Otdr.PhysicalFiberCharacteristics" );

                lElement.mLinkElement = New<DotNetInstance>( &lBaseLinkElementClass, lBaseLinkElementClass.Class(), x.mLinkElement );
                lElement.mPhysicalEvent =
                    New<DotNetInstance>( &lOlmPhysicalEventClass, lOlmPhysicalEventClass.Class(), x.mPhysicalEvent );
                lElement.mPeakTrace = New<DotNetInstance>( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), x.mPeakTrace );
                lElement.mDetectionTrace =
                    New<DotNetInstance>( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), x.mDetectionTrace );
                lElement.mAttributes = New<DotNetInstance>( &lOlmAttributeClass, lOlmAttributeClass.Class(), x.mAttributes );
                lElement.mAcquisitionData =
                    New<DotNetInstance>( &lAcquisitionDataClassType, lAcquisitionDataClassType.Class(), x.mAcquisitionData );
                lElement.mFiberInfo = New<DotNetInstance>( &lFiberInfoClassType, lFiberInfoClassType.Class(), x.mFiberInfo );
            }

            mLinkElementTable0->SetData( mLinkElementVector0 );

            mTracePlot->Clear();
            mTracePlot->SetEventData( mLinkElementVector0 );
        }

        {
            auto        lReanalyze       = true;
            MonoObject *lLinkElementData = mDataInstance->CallMethod( "GetLinkElements", &lReanalyze );

            auto lLinkElementVector = DotNetRuntime::AsVector<sDotNetLinkElement>( lLinkElementData );

            mLinkElementVector1 = std::vector<sLinkElement>();

            for( auto const &x : lLinkElementVector )
            {
                auto &lElement = mLinkElementVector1.emplace_back();

                lElement.mRowIndex            = x.mRowIndex;
                lElement.mLinkIndex           = x.mLinkIndex;
                lElement.mSubLinkIndex        = x.mSubLinkIndex;
                lElement.mEventIndex          = x.mEventIndex;
                lElement.mSubEventIndex       = x.mSubEventIndex;
                lElement.mIsSubElement        = x.mIsSubElement;
                lElement.mDiagnosicCount      = x.mDiagnosicCount;
                lElement.mLossPassFail        = x.mLossPassFail;
                lElement.mReflectancePassFail = x.mReflectancePassFail;

                static auto &lBaseLinkElementClass  = DotNetRuntime::GetClassType( "Metrino.Olm.BaseLinkElement" );
                static auto &lOlmPhysicalEventClass = DotNetRuntime::GetClassType( "Metrino.Olm.OlmPhysicalEvent" );
                static auto &lOlmAttributeClass =
                    DotNetRuntime::GetClassType( "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute" );
                static auto &lAcquisitionDataClassType = DotNetRuntime::GetClassType( "Metrino.Otdr.AcquisitionData" );
                static auto &lSinglePulseTraceClass    = DotNetRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );
                static auto &lFiberInfoClassType       = DotNetRuntime::GetClassType( "Metrino.Otdr.PhysicalFiberCharacteristics" );

                lElement.mLinkElement = New<DotNetInstance>( &lBaseLinkElementClass, lBaseLinkElementClass.Class(), x.mLinkElement );
                lElement.mPhysicalEvent =
                    New<DotNetInstance>( &lOlmPhysicalEventClass, lOlmPhysicalEventClass.Class(), x.mPhysicalEvent );
                lElement.mPeakTrace = New<DotNetInstance>( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), x.mPeakTrace );
                lElement.mDetectionTrace =
                    New<DotNetInstance>( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), x.mDetectionTrace );
                lElement.mAttributes = New<DotNetInstance>( &lOlmAttributeClass, lOlmAttributeClass.Class(), x.mAttributes );
                lElement.mAcquisitionData =
                    New<DotNetInstance>( &lAcquisitionDataClassType, lAcquisitionDataClassType.Class(), x.mAcquisitionData );
                lElement.mFiberInfo = New<DotNetInstance>( &lFiberInfoClassType, lFiberInfoClassType.Class(), x.mFiberInfo );
            }

            mLinkElementTable1->SetData( mLinkElementVector1 );
        }
    }

} // namespace SE::OtdrEditor