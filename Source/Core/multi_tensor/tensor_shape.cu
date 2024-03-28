#include "tensor_shape.h"

namespace SE::Core
{
    tensor_shape_t::tensor_shape_t( vector_t<vector_t<uint32_t>> const &shape, size_t elementSize )
    {
        if( shape.size() == 0 )
            return;

        for( auto &x : shape )
        {
            if( x.size() != shape[0].size() )
                throw std::runtime_error( "All shapes must have the same length!" );
        }

        Rank        = shape[0].size();
        LayerCount  = shape.size();
        Shape       = shape;
        ElementSize = elementSize;

        UpdateMetadata();
    }

    tensor_shape_t::tensor_shape_t( vector_t<uint32_t> const &shape, size_t elementSize )
    {
        if( shape.size() == 0 )
            return;

        Rank       = 1;
        LayerCount = shape.size();

        for( auto &lValue : shape )
            Shape.push_back( { lValue } );

        ElementSize = elementSize;

        UpdateMetadata();
    }

    void tensor_shape_t::UpdateMetadata()
    {
        MaxBufferSize = 0;

        MaxDimensions.resize( Rank );
        std::fill( MaxDimensions.begin(), MaxDimensions.end(), 0 );

        BufferSizes.resize( Shape.size() );
        Strides.resize( Shape.size() );

        size_t currentOffset = 0;
        for( size_t dimIdx = 0; dimIdx < Shape.size(); dimIdx++ )
        {
            auto &lDim = Shape[dimIdx];

            Strides[dimIdx]            = vector_t<uint32_t>( Rank );
            Strides[dimIdx][Rank - 1] = 1;

            uint32_t size = ElementSize;
            for( uint32_t i = 0; i < Rank; i++ )
            {
                MaxDimensions[i] = std::max( MaxDimensions[i], lDim[i] );
                if( i < Rank - 1 )
                    Strides[dimIdx][Rank - i - 2] = Strides[dimIdx][Rank - i - 1] * lDim[Rank - i - 1];

                size *= lDim[i];
            }

            BufferSizes[dimIdx].Size   = size;
            BufferSizes[dimIdx].Offset = currentOffset;
            MaxBufferSize                = std::max( MaxBufferSize, BufferSizes[dimIdx].Size / ElementSize );
            currentOffset += BufferSizes[dimIdx].Size;
        }

        ByteSize = currentOffset;
    }

    vector_t<uint32_t> const tensor_shape_t::GetDimension( int32_t i ) const
    {
        vector_t<uint32_t> dimension;

        if( i >= 0 )
        {
            if( i >= Rank )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", Rank + i, CountLayers() ) );
            for( auto &lShape : Shape )
                dimension.push_back( lShape[i] );
        }
        else
        {
            if( -i > Rank )
                throw std::out_of_range(
                    fmt::format( "Attempted to access layer {}, but the stack only has {} layers", Rank + i, CountLayers() ) );

            for( auto &lShape : Shape )
                dimension.push_back( lShape[Rank + i] );
        }
        return dimension;
    }

    void tensor_shape_t::InsertDimension( int32_t position, vector_t<uint32_t> dimension )
    {
        if( dimension.size() != CountLayers() )
            throw std::out_of_range(
                fmt::format( "New dimension array has size {}, but the tensor has {} layers", dimension.size(), CountLayers() ) );

        if( position < 0 )
            position += ( Rank + 1 );

        for( uint32_t i = 0; i < CountLayers(); i++ )
            Shape[i].insert( Shape[i].begin() + position, dimension[i] );

        Rank++;

        UpdateMetadata();
    }

    void tensor_shape_t::Flatten( int32_t toDimension )
    {
        if( toDimension <= 0 )
            toDimension += Rank;

        vector_t<vector_t<uint32_t>> newShape( CountLayers() );

        for( uint32_t i = 0; i < CountLayers(); i++ )
        {
            newShape[i].push_back(
                std::accumulate( Shape[i].begin(), Shape[i].begin() + toDimension, 1, std::multiplies<uint32_t>() ) );
            newShape[i].insert( newShape[i].end(), Shape[i].begin() + toDimension, Shape[i].end() );
        }

        Shape = newShape;
        Rank  = Rank - toDimension + 1;

        UpdateMetadata();
    }

    void tensor_shape_t::Trim( int32_t toDimension )
    {
        if( toDimension == 0 )
            return;

        if( toDimension < 0 )
            toDimension += Rank;

        vector_t<vector_t<uint32_t>> newShape( CountLayers() );

        for( uint32_t i = 0; i < CountLayers(); i++ )
            newShape[i].insert( newShape[i].end(), Shape[i].begin(), Shape[i].begin() + toDimension );

        Shape = newShape;
        Rank  = toDimension;

        UpdateMetadata();
    }

    bool operator==( const buffer_size_info_t &lhs, const buffer_size_info_t &rhs )
    {
        return ( lhs.Size == rhs.Size ) && ( lhs.Offset == rhs.Offset );
    }

    bool tensor_shape_t::operator==( const tensor_shape_t &rhs )
    {
        return ( Shape == rhs.Shape );
    }

    bool tensor_shape_t::operator!=( const tensor_shape_t &rhs )
    {
        return ( Shape != rhs.Shape );
    }

    void tensor_shape_t::SyncDeviceData()
    {
        vector_t<uint32_t> dimensions( LayerCount * Rank );

        uint32_t k = 0;
        for( uint32_t i = 0; i < LayerCount; i++ )
        {
            for( uint32_t j = 0; j < Rank; j++ )
            {
                dimensions[k] = Shape[i][j];
                k++;
            }
        }
        DeviceSideData.Shape.Upload( dimensions );
        DeviceSideData.MaxDimensions.Upload( MaxDimensions );
        DeviceSideData.BufferSizes.Upload( BufferSizes );
    }
}