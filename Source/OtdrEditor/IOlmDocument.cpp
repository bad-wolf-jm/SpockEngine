#include "IOlmDocument.h"

#include "DotNet/Runtime.h"

namespace SE::OtdrEditor
{
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

            mLinkElementVector = DotNetRuntime::AsVector<sLinkElement>( lLinkElementData );
            mLinkElementTable->SetData( mLinkElementVector );
            mTracePlot->Clear();
            mTracePlot->SetEventData( mLinkElementVector );
        }
    }

} // namespace SE::OtdrEditor