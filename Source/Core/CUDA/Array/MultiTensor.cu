/// @file   MultiTensor.cu
///
/// @brief  MultiTensor class implementation
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#include "MultiTensor.h"
#include <stdexcept>

namespace SE::Cuda
{
    sTensorShape::sTensorShape( std::vector<std::vector<uint32_t>> const &aShape, size_t aElementSize )
    {
        if( aShape.size() == 0 ) return;

        for( auto &x : aShape )
        {
            if( x.size() != aShape[0].size() ) throw std::runtime_error( "All shapes must have the same length!" );
        }

        mRank        = aShape[0].size();
        mLayerCount  = aShape.size();
        mShape       = aShape;
        mElementSize = aElementSize;

        UpdateMetadata();
    }

    sTensorShape::sTensorShape( std::vector<uint32_t> const &aShape, size_t aElementSize )
    {
        if( aShape.size() == 0 ) return;

        mRank       = 1;
        mLayerCount = aShape.size();

        for( auto &lValue : aShape ) mShape.push_back( { lValue } );

        mElementSize = aElementSize;

        UpdateMetadata();
    }

    void sTensorShape::UpdateMetadata()
    {
        mMaxBufferSize = 0;

        mMaxDimensions.resize( mRank );
        std::fill( mMaxDimensions.begin(), mMaxDimensions.end(), 0 );

        mBufferSizes.resize( mShape.size() );
        mStrides.resize( mShape.size() );

        size_t lCurrentOffset = 0;
        for( size_t lDimIdx = 0; lDimIdx < mShape.size(); lDimIdx++ )
        {
            auto &lDim = mShape[lDimIdx];

            mStrides[lDimIdx]            = std::vector<uint32_t>( mRank );
            mStrides[lDimIdx][mRank - 1] = 1;

            uint32_t lSize = mElementSize;
            for( uint32_t i = 0; i < mRank; i++ )
            {
                mMaxDimensions[i] = std::max( mMaxDimensions[i], lDim[i] );
                if( i < mRank - 1 ) mStrides[lDimIdx][mRank - i - 2] = mStrides[lDimIdx][mRank - i - 1] * lDim[mRank - i - 1];
                lSize *= lDim[i];
            }

            mBufferSizes[lDimIdx].mSize   = lSize;
            mBufferSizes[lDimIdx].mOffset = lCurrentOffset;
            mMaxBufferSize                = std::max( mMaxBufferSize, mBufferSizes[lDimIdx].mSize / mElementSize );
            lCurrentOffset += mBufferSizes[lDimIdx].mSize;
        }

        mByteSize = lCurrentOffset;
    }

    std::vector<uint32_t> const sTensorShape::GetDimension( int32_t i ) const
    {
        std::vector<uint32_t> lDimension;

        if( i >= 0 )
        {
            if( i >= mRank )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", mRank + i, CountLayers() ) );
            for( auto &lShape : mShape ) lDimension.push_back( lShape[i] );
        }
        else
        {
            if( -i > mRank )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", mRank + i, CountLayers() ) );

            for( auto &lShape : mShape ) lDimension.push_back( lShape[mRank + i] );
        }
        return lDimension;
    }

    void sTensorShape::InsertDimension( int32_t aPosition, std::vector<uint32_t> aDimension )
    {
        if( aDimension.size() != CountLayers() )
            throw std::out_of_range(
                fmt::format( "New dimension array has size {}, but the tensor has {} layers", aDimension.size(), CountLayers() ) );

        if( aPosition < 0 ) aPosition += ( mRank + 1 );

        for( uint32_t i = 0; i < CountLayers(); i++ ) mShape[i].insert( mShape[i].begin() + aPosition, aDimension[i] );

        mRank++;

        UpdateMetadata();
    }

    void sTensorShape::Flatten( int32_t aToDimension )
    {
        if( aToDimension <= 0 ) aToDimension += mRank;

        std::vector<std::vector<uint32_t>> lNewShape( CountLayers() );

        for( uint32_t i = 0; i < CountLayers(); i++ )
        {
            lNewShape[i].push_back(
                std::accumulate( mShape[i].begin(), mShape[i].begin() + aToDimension, 1, std::multiplies<uint32_t>() ) );
            lNewShape[i].insert( lNewShape[i].end(), mShape[i].begin() + aToDimension, mShape[i].end() );
        }

        mShape = lNewShape;
        mRank  = mRank - aToDimension + 1;

        UpdateMetadata();
    }

    void sTensorShape::Trim( int32_t aToDimension )
    {
        if( aToDimension == 0 ) return;

        if( aToDimension < 0 ) aToDimension += mRank;

        std::vector<std::vector<uint32_t>> lNewShape( CountLayers() );

        for( uint32_t i = 0; i < CountLayers(); i++ )
            lNewShape[i].insert( lNewShape[i].end(), mShape[i].begin(), mShape[i].begin() + aToDimension );

        mShape = lNewShape;
        mRank  = aToDimension;

        UpdateMetadata();
    }

    bool operator==( const sBufferSizeInfo &lLhs, const sBufferSizeInfo &lRhs )
    {
        return ( lLhs.mSize == lRhs.mSize ) && ( lLhs.mOffset == lRhs.mOffset );
    }

    bool sTensorShape::operator==( const sTensorShape &lRhs ) { return ( mShape == lRhs.mShape ); }

    bool sTensorShape::operator!=( const sTensorShape &lRhs ) { return ( mShape != lRhs.mShape ); }

    void sTensorShape::SyncDeviceData()
    {
        std::vector<uint32_t> lDimensions( mLayerCount * mRank );
        uint32_t              k = 0;
        for( uint32_t i = 0; i < mLayerCount; i++ )
        {
            for( uint32_t j = 0; j < mRank; j++ )
            {
                lDimensions[k] = mShape[i][j];
                k++;
            }
        }
        mDeviceSideData.mShape.Upload( lDimensions );
        mDeviceSideData.mMaxDimensions.Upload( mMaxDimensions );
        mDeviceSideData.mBufferSizes.Upload( mBufferSizes );
    }

    MultiTensor::MultiTensor( MemoryPool &aMemoryPool, const sTensorShape &aShape )
        : mShape{ aShape }
    {
        mMemoryBuffer                         = aMemoryPool.Allocate( mShape.mByteSize );
        mShape.mDeviceSideData.mShape         = aMemoryPool.Allocate( mShape.mLayerCount * mShape.mRank * sizeof( uint32_t ) );
        mShape.mDeviceSideData.mMaxDimensions = aMemoryPool.Allocate( mShape.mRank * sizeof( uint32_t ) );
        mShape.mDeviceSideData.mBufferSizes   = aMemoryPool.Allocate( mShape.mLayerCount * sizeof( sBufferSizeInfo ) );
        mShape.SyncDeviceData();
    }

    MultiTensor::MultiTensor( MemoryPool &aMemoryPool, MemoryBuffer &aMemoryBuffer, const sTensorShape &aShape )
        : mShape{ aShape }
    {
        mMemoryBuffer                         = aMemoryBuffer;
        mShape.mDeviceSideData.mShape         = aMemoryPool.Allocate( mShape.mLayerCount * mShape.mRank * sizeof( uint32_t ) );
        mShape.mDeviceSideData.mMaxDimensions = aMemoryPool.Allocate( mShape.mRank * sizeof( uint32_t ) );
        mShape.mDeviceSideData.mBufferSizes   = aMemoryPool.Allocate( mShape.mLayerCount * sizeof( sBufferSizeInfo ) );
        mShape.SyncDeviceData();
    }

} // namespace SE::Cuda