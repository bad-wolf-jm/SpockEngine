#pragma once

#include "Core/Memory.h"
#include "Optix7.h"
#include "OptixProgramGroup.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    struct OptixShaderBindingTableObject
    {
        OptixShaderBindingTable mOptixObject;

        OptixShaderBindingTableObject() = default;

        ~OptixShaderBindingTableObject() = default;

        template <typename _RecordType>
        _RecordType NewRecordType( Ref<SE::Graphics::OptixProgramGroupObject> aProgramGroup )
        {
            _RecordType lNewRecord;
            OPTIX_CHECK( optixSbtRecordPackHeader( aProgramGroup->mOptixObject, &lNewRecord ) );
            return lNewRecord;
        }

        template <typename _RecordType>
        std::vector<_RecordType> NewRecordType( std::vector<Ref<SE::Graphics::OptixProgramGroupObject>> aProgramGroup )
        {
            std::vector<_RecordType> lNewRecordTypes;
            for( int i = 0; i < aProgramGroup.size(); i++ )
            {
                _RecordType rec = NewRecordType<_RecordType>( aProgramGroup[i] );
                lNewRecordTypes.push_back( rec );
            }
            return lNewRecordTypes;
        }

        void BindRayGenRecordTable( CUdeviceptr aDevicePointer ) { mOptixObject.raygenRecord = aDevicePointer; }

        template <typename _RecordType>
        void BindMissRecordTable( CUdeviceptr aDevicePointer, size_t aSize )
        {
            mOptixObject.missRecordBase          = aDevicePointer;
            mOptixObject.missRecordStrideInBytes = sizeof( _RecordType );
            mOptixObject.missRecordCount         = static_cast<int>( aSize );
        }

        template <typename _RecordType>
        void BindHitRecordTable( CUdeviceptr aDevicePointer, size_t aSize )
        {
            mOptixObject.hitgroupRecordBase          = aDevicePointer;
            mOptixObject.hitgroupRecordStrideInBytes = sizeof( _RecordType );
            mOptixObject.hitgroupRecordCount         = static_cast<int>( aSize );
        }
    };

} // namespace SE::Graphics