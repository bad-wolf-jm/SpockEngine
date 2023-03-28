#include "IOlmDocument.h"

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

    UIIolmDocument::UIIolmDocument( fs::path aPath, bool aReanalyse )
    {
        mOpen  = true;
        mDirty = false;
        mName  = aPath.filename().string();

        mTracePlot        = New<UILinkElementTracePlot>();
        mLinkElementTable = New<UILinkElementTable>();

        mMainLayout = New<UIBoxLayout>( eBoxLayoutOrientation::VERTICAL );
        mMainLayout->Add( mLinkElementTable.get(), true, true );
        mMainLayout->Add( mTracePlot.get(), true, true );

        SetContent( mMainLayout.get() );

        mLinkElementTable->OnElementClicked(
            [&]( sLinkElement const &aElement )
            {
                auto lWavelength = aElement.mPhysicalEvent->GetPropertyValue<double>( "Wavelength" ) * 1e9;
                std::string lTitle = fmt::format("Event Trace: {} nm", lWavelength);
                mTracePlot->SetTitle(lTitle);
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

        {
            MonoObject *lLinkElementData = mDataInstance->CallMethod( "GetLinkElements", &aReanalyse );

            auto lLinkElementVector = DotNetRuntime::AsVector<sDotNetLinkElement>( lLinkElementData );

            mLinkElementVector = std::vector<sLinkElement>();

            for( auto const &x : lLinkElementVector )
            {
                auto &lElement = mLinkElementVector.emplace_back();

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

            mLinkElementTable->SetData( mLinkElementVector );
            mTracePlot->Clear();
            mTracePlot->SetEventData( mLinkElementVector );
        }
    }

} // namespace SE::OtdrEditor