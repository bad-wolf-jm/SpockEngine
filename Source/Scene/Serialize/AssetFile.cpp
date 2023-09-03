#include "AssetFile.h"
#include "Core/Logging.h"
namespace SE::Core
{
    static constexpr uint8_t sFileMagic[] = { '@', '%', 'S', 'P', 'O', 'C', 'K', 'E', 'N', 'G', 'I', 'N', 'E', '_',
                                              'S', 'C', 'E', 'N', 'E', '_', 'A', 'S', 'S', 'E', 'T', '@', '%' };

    BinaryAsset::BinaryAsset( fs::path const &aPath )
        : mFilePath{ aPath }
        , mFileExists{ false }
        , mFileSize{ 0 }
    {
        mFileStream.open( mFilePath, std::ifstream::binary );
        mFileExists = mFileStream.good();
        mFileSize   = 0;

        if( !mFileExists )
            throw std::runtime_error( "Specified file does not exist" );

        mFileStream.seekg( 0, mFileStream.end );
        mFileSize = mFileStream.tellg();
        mFileStream.seekg( 0, mFileStream.beg );

        if( mFileSize < sizeof( sFileMagic ) )
            throw std::runtime_error( "Wrong file type!!!" );

        std::vector<uint8_t> lMagic = Read<uint8_t>( sizeof( sFileMagic ) );
        if( memcmp( lMagic.data(), sFileMagic, sizeof( sFileMagic ) != 0 ) )
            throw std::runtime_error( "Magic value does not match!!!" );

        mAssetCount = Read<uint32_t>();
        mAssetIndex = Read<sAssetIndex>( mAssetCount );
    }

    BinaryAsset::~BinaryAsset()
    {
        mFileStream.close();
    }

    const uint8_t *BinaryAsset::GetMagic()
    {
        return sFileMagic;
    }
    const uint32_t BinaryAsset::GetMagicLength()
    {
        return static_cast<uint32_t>( sizeof( sFileMagic ) );
    }

    size_t BinaryAsset::CurrentPosition()
    {
        return mFileStream.tellg();
    }

    void BinaryAsset::Seek( size_t aPosition )
    {
        mFileStream.seekg( aPosition, mFileStream.beg );
    }

    bool BinaryAsset::Eof()
    {
        return mFileStream.eof();
    }

    sAssetIndex const &BinaryAsset::GetIndex( uint32_t aIndex ) const
    {
        return mAssetIndex[aIndex];
    }

    void BinaryAsset::WriteTo( fs::path aPath )
    {

        uint32_t lDataStartOffset = BinaryAsset::GetMagicLength() + sizeof( uint32_t ) + mAssetIndex.size() * sizeof( sAssetIndex );
        for( uint32_t i = 0; i < mAssetIndex.size(); i++ )
        {
            mAssetIndex[i].mByteStart += lDataStartOffset;
            mAssetIndex[i].mByteEnd += lDataStartOffset;
        }

        auto lOutFile = std::ofstream( aPath.string(), std::ofstream::binary );

        auto *lMagic       = BinaryAsset::GetMagic();
        auto  lMagicLength = BinaryAsset::GetMagicLength();
        lOutFile.write( (const char *)lMagic, lMagicLength );

        uint32_t lAssetCount = static_cast<uint32_t>( mAssetIndex.size() );
        lOutFile.write( (const char *)&lAssetCount, sizeof( uint32_t ) );

        lOutFile.write( (const char *)mAssetIndex.data(), mAssetIndex.size() * sizeof( sAssetIndex ) );
        for( auto &lPacket : mPackets )
            lOutFile.write( (const char *)lPacket.data(), lPacket.size() );
    }

    void BinaryAsset::Package( TextureData2D const &aData, sTextureSamplingInfo const &aSampler )
    {
        uint32_t lHeaderSize = 0;
        lHeaderSize += sizeof( eSamplerFilter ) + sizeof( eSamplerFilter ) + sizeof( eSamplerMipmap ) + sizeof( eSamplerWrapping );
        lHeaderSize += 2 * sizeof( float );
        lHeaderSize += 2 * sizeof( float );
        lHeaderSize += 4 * sizeof( float );

        auto     lKTXData    = aData.Serialize();
        uint32_t lPacketSize = lKTXData.size() + lHeaderSize;

        mPackets.emplace_back( lPacketSize );

        auto *lPtr = mPackets.back().data();
        std::memcpy( lPtr, &aSampler.mFilter, sizeof( eSamplerFilter ) );
        lPtr += sizeof( eSamplerFilter );
        std::memcpy( lPtr, &aSampler.mFilter, sizeof( eSamplerFilter ) );
        lPtr += sizeof( eSamplerFilter );
        std::memcpy( lPtr, &aSampler.mMipFilter, sizeof( eSamplerMipmap ) );
        lPtr += sizeof( eSamplerMipmap );
        std::memcpy( lPtr, &aSampler.mWrapping, sizeof( eSamplerWrapping ) );
        lPtr += sizeof( eSamplerWrapping );

        std::memcpy( lPtr, &aSampler.mScaling[0], sizeof( float ) );
        lPtr += sizeof( float );
        std::memcpy( lPtr, &aSampler.mScaling[1], sizeof( float ) );
        lPtr += sizeof( float );

        std::memcpy( lPtr, &aSampler.mOffset[0], sizeof( float ) );
        lPtr += sizeof( float );
        std::memcpy( lPtr, &aSampler.mOffset[1], sizeof( float ) );
        lPtr += sizeof( float );

        std::memcpy( lPtr, &aSampler.mBorderColor[0], sizeof( float ) );
        lPtr += sizeof( float );
        std::memcpy( lPtr, &aSampler.mBorderColor[1], sizeof( float ) );
        lPtr += sizeof( float );
        std::memcpy( lPtr, &aSampler.mBorderColor[2], sizeof( float ) );
        lPtr += sizeof( float );
        std::memcpy( lPtr, &aSampler.mBorderColor[3], sizeof( float ) );
        lPtr += sizeof( float );

        std::memcpy( lPtr, lKTXData.data(), lKTXData.size() );

        sAssetIndex lIndexEntry{};
        lIndexEntry.mType      = eAssetType::KTX_TEXTURE_2D;
        lIndexEntry.mByteStart = mTotalPacketSize;
        lIndexEntry.mByteEnd   = mTotalPacketSize + lPacketSize;
        mAssetIndex.push_back( lIndexEntry );
        mTotalPacketSize += lPacketSize;
    }

    void BinaryAsset::Package( ref_t<TextureData2D> aData, ref_t<TextureSampler2D> aSampler )
    {
        if( aData != nullptr )
        {
            return Package( *aData, aSampler->mSamplingSpec );
        }
        else
        {
            return Package( TextureData2D{}, sTextureSamplingInfo{} );
        }
    }

    void BinaryAsset::Package( sImportedTexture const &aData )
    {
        return Package( aData.mTexture, aData.mSampler );
    }

    std::tuple<TextureData2D, TextureSampler2D> BinaryAsset::Retrieve( uint32_t aIndex )
    {
        auto lAssetIndex = mAssetIndex[aIndex];
        if( lAssetIndex.mType != eAssetType::KTX_TEXTURE_2D )
            throw std::runtime_error( "Binary data type mismatch" );

        Seek( lAssetIndex.mByteStart );

        sTextureSamplingInfo lSamplerCreateInfo{};
        lSamplerCreateInfo.mFilter                = Read<eSamplerFilter>();
        lSamplerCreateInfo.mFilter                = Read<eSamplerFilter>();
        lSamplerCreateInfo.mMipFilter             = Read<eSamplerMipmap>();
        lSamplerCreateInfo.mWrapping              = Read<eSamplerWrapping>();
        lSamplerCreateInfo.mNormalizedCoordinates = true;
        lSamplerCreateInfo.mNormalizedValues      = true;
        lSamplerCreateInfo.mScaling[0]            = Read<float>();
        lSamplerCreateInfo.mScaling[1]            = Read<float>();
        lSamplerCreateInfo.mOffset[0]             = Read<float>();
        lSamplerCreateInfo.mOffset[1]             = Read<float>();
        lSamplerCreateInfo.mBorderColor[0]        = Read<float>();
        lSamplerCreateInfo.mBorderColor[1]        = Read<float>();
        lSamplerCreateInfo.mBorderColor[2]        = Read<float>();
        lSamplerCreateInfo.mBorderColor[3]        = Read<float>();

        uint32_t lKTXDataSize = lAssetIndex.mByteEnd - static_cast<uint32_t>( CurrentPosition() );

        auto lData = Read<char>( lKTXDataSize );

        TextureData2D    lTextureData( lData.data(), lData.size() );
        TextureSampler2D lSampler( lTextureData, lSamplerCreateInfo );

        return { lTextureData, lSampler };
    }

    void BinaryAsset::Package( std::vector<VertexData> const &aVertexData, std::vector<uint32_t> const &aIndexData )
    {
        uint32_t lHeaderSize = 2 * sizeof( uint32_t );
        uint32_t lPacketSize = aVertexData.size() * sizeof( VertexData ) + aIndexData.size() * sizeof( uint32_t ) + lHeaderSize;

        mPackets.emplace_back( lPacketSize );
        uint32_t lVertexByteSize = aVertexData.size();
        uint32_t lIndexByteSize  = aIndexData.size();

        auto *lPtr = mPackets.back().data();
        std::memcpy( lPtr, &lVertexByteSize, sizeof( uint32_t ) );
        lPtr += sizeof( uint32_t );
        std::memcpy( lPtr, &lIndexByteSize, sizeof( uint32_t ) );
        lPtr += sizeof( uint32_t );
        std::memcpy( lPtr, aVertexData.data(), aVertexData.size() * sizeof( VertexData ) );
        lPtr += aVertexData.size() * sizeof( VertexData );
        std::memcpy( lPtr, aIndexData.data(), aIndexData.size() * sizeof( uint32_t ) );

        sAssetIndex lIndexEntry{};
        lIndexEntry.mType      = eAssetType::MESH_DATA;
        lIndexEntry.mByteStart = mTotalPacketSize;
        lIndexEntry.mByteEnd   = mTotalPacketSize + lPacketSize;
        mAssetIndex.push_back( lIndexEntry );
        mTotalPacketSize += lPacketSize;
    }

    void BinaryAsset::Retrieve( uint32_t aIndex, std::vector<VertexData> &aVertexData, std::vector<uint32_t> &aIndexData )
    {
        auto lAssetIndex = mAssetIndex[aIndex];
        if( lAssetIndex.mType != eAssetType::MESH_DATA )
            throw std::runtime_error( "Binary data type mismatch" );

        Seek( lAssetIndex.mByteStart );

        auto lVertexBufferSize = Read<uint32_t>();
        auto lIndexBufferSize  = Read<uint32_t>();

        aVertexData = Read<VertexData>( lVertexBufferSize );
        aIndexData  = Read<uint32_t>( lIndexBufferSize );
    }

    void BinaryAsset::Package( sMaterial const &aMaterialData )
    {
        uint32_t lHeaderSize = sizeof( uint32_t );
        uint32_t lPacketSize = sizeof( sMaterial ) - sizeof( string_t ) + ( sizeof( uint32_t ) + aMaterialData.mName.size() );

        mPackets.emplace_back( lPacketSize );

        auto    *lPtr          = mPackets.back().data();
        uint32_t lNameByteSize = aMaterialData.mName.size();

        std::memcpy( lPtr, &lNameByteSize, sizeof( uint32_t ) );
        lPtr += sizeof( uint32_t );

        std::memcpy( lPtr, aMaterialData.mName.c_str(), lNameByteSize );
        lPtr += lNameByteSize;

        std::memcpy( lPtr, &aMaterialData.mID, sizeof( uint32_t ) );
        lPtr += sizeof( uint32_t );

        std::memcpy( lPtr, &aMaterialData.mType, sizeof( uint8_t ) );
        lPtr += sizeof( uint8_t );

        std::memcpy( lPtr, &aMaterialData.mLineWidth, sizeof( float ) );
        lPtr += sizeof( float );
        std::memcpy( lPtr, &aMaterialData.mIsTwoSided, sizeof( bool ) );
        lPtr += sizeof( bool );
        std::memcpy( lPtr, &aMaterialData.mUseAlphaMask, sizeof( bool ) );
        lPtr += sizeof( bool );
        std::memcpy( lPtr, &aMaterialData.mAlphaThreshold, sizeof( float ) );
        lPtr += sizeof( float );

        std::memcpy( lPtr, &aMaterialData.mBaseColorFactor, sizeof( math::vec4 ) );
        lPtr += sizeof( math::vec4 );

        std::memcpy( lPtr, &aMaterialData.mBaseColorTexture, sizeof( sTextureReference ) );
        lPtr += sizeof( sTextureReference );

        std::memcpy( lPtr, &aMaterialData.mEmissiveFactor, sizeof( math::vec4 ) );
        lPtr += sizeof( math::vec4 );

        std::memcpy( lPtr, &aMaterialData.mEmissiveTexture, sizeof( sTextureReference ) );
        lPtr += sizeof( sTextureReference );

        std::memcpy( lPtr, &aMaterialData.mRoughnessFactor, sizeof( float ) );
        lPtr += sizeof( float );

        std::memcpy( lPtr, &aMaterialData.mMetallicFactor, sizeof( float ) );
        lPtr += sizeof( float );

        std::memcpy( lPtr, &aMaterialData.mMetalRoughTexture, sizeof( sTextureReference ) );
        lPtr += sizeof( sTextureReference );

        std::memcpy( lPtr, &aMaterialData.mOcclusionStrength, sizeof( float ) );
        lPtr += sizeof( float );

        std::memcpy( lPtr, &aMaterialData.mOcclusionTexture, sizeof( sTextureReference ) );
        lPtr += sizeof( sTextureReference );

        std::memcpy( lPtr, &aMaterialData.mNormalsTexture, sizeof( sTextureReference ) );
        lPtr += sizeof( sTextureReference );

        sAssetIndex lIndexEntry{};
        lIndexEntry.mType      = eAssetType::MATERIAL_DATA;
        lIndexEntry.mByteStart = mTotalPacketSize;
        lIndexEntry.mByteEnd   = mTotalPacketSize + lPacketSize;
        mAssetIndex.push_back( lIndexEntry );
        mTotalPacketSize += lPacketSize;
    }

    void BinaryAsset::Retrieve( uint32_t aIndex, sMaterial &aMaterialData )
    {
        auto lAssetIndex = mAssetIndex[aIndex];
        if( lAssetIndex.mType != eAssetType::MATERIAL_DATA )
            throw std::runtime_error( "Binary data type mismatch" );

        Seek( lAssetIndex.mByteStart );

        aMaterialData.mName              = Read<string_t>();
        aMaterialData.mID                = Read<uint32_t>();
        aMaterialData.mType              = Read<eMaterialType>();
        aMaterialData.mLineWidth         = Read<float>();
        aMaterialData.mIsTwoSided        = Read<bool>();
        aMaterialData.mUseAlphaMask      = Read<bool>();
        aMaterialData.mAlphaThreshold    = Read<float>();
        aMaterialData.mBaseColorFactor   = Read<math::vec4>();
        aMaterialData.mBaseColorTexture  = Read<sTextureReference>();
        aMaterialData.mEmissiveFactor    = Read<math::vec4>();
        aMaterialData.mEmissiveTexture   = Read<sTextureReference>();
        aMaterialData.mRoughnessFactor   = Read<float>();
        aMaterialData.mMetallicFactor    = Read<float>();
        aMaterialData.mMetalRoughTexture = Read<sTextureReference>();
        aMaterialData.mOcclusionStrength = Read<float>();
        aMaterialData.mOcclusionTexture  = Read<sTextureReference>();
        aMaterialData.mNormalsTexture    = Read<sTextureReference>();
    }

    void BinaryAsset::Package( sImportedAnimationSampler const &aMaterialData )
    {
        uint32_t lHeaderSize = sizeof( uint32_t ) + sizeof( uint8_t );
        uint32_t lPacketSize =
            lHeaderSize + aMaterialData.mInputs.size() * sizeof( float ) + aMaterialData.mOutputsVec4.size() * sizeof( math::vec4 );

        mPackets.emplace_back( lPacketSize );

        auto *lPtr = mPackets.back().data();

        std::memcpy( lPtr, &aMaterialData.mInterpolation, sizeof( uint8_t ) );
        lPtr += sizeof( uint8_t );

        uint32_t lInterpolationLength = aMaterialData.mInputs.size();
        std::memcpy( lPtr, &lInterpolationLength, sizeof( uint32_t ) );
        lPtr += sizeof( uint32_t );

        std::memcpy( lPtr, aMaterialData.mInputs.data(), aMaterialData.mInputs.size() * sizeof( float ) );
        lPtr += aMaterialData.mInputs.size() * sizeof( float );

        uint32_t lOutputLength = aMaterialData.mOutputsVec4.size();
        std::memcpy( lPtr, &lOutputLength, sizeof( uint32_t ) );
        lPtr += sizeof( uint32_t );

        std::memcpy( lPtr, aMaterialData.mOutputsVec4.data(), aMaterialData.mOutputsVec4.size() * sizeof( math::vec4 ) );
        lPtr += aMaterialData.mOutputsVec4.size() * sizeof( math::vec4 );

        sAssetIndex lIndexEntry{};
        lIndexEntry.mType      = eAssetType::ANIMATION_DATA;
        lIndexEntry.mByteStart = mTotalPacketSize;
        lIndexEntry.mByteEnd   = mTotalPacketSize + lPacketSize;
        mAssetIndex.push_back( lIndexEntry );
        mTotalPacketSize += lPacketSize;
    }

    void BinaryAsset::Retrieve( uint32_t aIndex, sImportedAnimationSampler &aMaterialData )
    {
        auto lAssetIndex = mAssetIndex[aIndex];
        if( lAssetIndex.mType != eAssetType::ANIMATION_DATA )
            throw std::runtime_error( "Binary data type mismatch" );

        Seek( lAssetIndex.mByteStart );

        aMaterialData.mInterpolation = Read<sImportedAnimationSampler::Interpolation>();

        auto lInputSize       = Read<uint32_t>();
        aMaterialData.mInputs = Read<float>( lInputSize );

        auto lOutputSize           = Read<uint32_t>();
        aMaterialData.mOutputsVec4 = Read<math::vec4>( lOutputSize );

        SE::Logging::Info( "{}", aIndex );
    }

} // namespace SE::Core