#include "AssetFile.h"
#include "Core/Logging.h"
namespace LTSE::Core
{
    static constexpr uint8_t sFileMagic[] = { '@', '%', 'S', 'P', 'O', 'C', 'K', 'E', 'N', 'G', 'I', 'N', 'E', '_', 'S', 'C', 'E', 'N',
        'E', '_', 'A', 'S', 'S', 'E', 'T', '@', '%' };

    BinaryAsset::BinaryAsset( fs::path const &aPath )
        : mFilePath{ aPath }
        , mFileExists{ false }
        , mFileSize{ 0 }
    {
        mFileStream.open( mFilePath, std::ifstream::binary );
        mFileExists = mFileStream.good();
        mFileSize   = 0;

        if( !mFileExists ) throw std::runtime_error( "Specified file does not exist" );

        mFileStream.seekg( 0, mFileStream.end );
        mFileSize = mFileStream.tellg();
        mFileStream.seekg( 0, mFileStream.beg );

        if( mFileSize < sizeof( sFileMagic ) ) throw std::runtime_error( "Wrong file type!!!" );

        std::vector<uint8_t> lMagic = Read<uint8_t>( sizeof( sFileMagic ) );
        if( memcmp( lMagic.data(), sFileMagic, sizeof( sFileMagic ) != 0 ) )
            throw std::runtime_error( "Magic value does not match!!!" );

        mAssetCount = Read<uint32_t>();
        mAssetIndex = Read<sAssetIndex>( mAssetCount );
    }

    BinaryAsset::~BinaryAsset() { mFileStream.close(); }

    const uint8_t *BinaryAsset::GetMagic() { return sFileMagic; }
    const uint32_t BinaryAsset::GetMagicLength() { return static_cast<uint32_t>( sizeof( sFileMagic ) ); }

    size_t BinaryAsset::CurrentPosition() { return mFileStream.tellg(); }

    void BinaryAsset::Seek( size_t aPosition ) { mFileStream.seekg( aPosition, mFileStream.beg ); }

    bool BinaryAsset::Eof() { return mFileStream.eof(); }

    sAssetIndex const &BinaryAsset::GetIndex( uint32_t aIndex ) const { return mAssetIndex[aIndex]; }

    std::vector<char> BinaryAsset::Package( TextureData2D const &aData, sTextureSamplingInfo const &aSampler )
    {
        uint32_t lHeaderSize = 0;
        lHeaderSize += sizeof( eSamplerFilter ) + sizeof( eSamplerFilter ) + sizeof( eSamplerMipmap ) + sizeof( eSamplerWrapping );
        lHeaderSize += 2 * sizeof( float );
        lHeaderSize += 2 * sizeof( float );
        lHeaderSize += 4 * sizeof( float );

        auto     lKTXData    = aData.Serialize();
        uint32_t lPacketSize = lKTXData.size() + lHeaderSize;

        std::vector<char> lPacket( lPacketSize );

        auto *lPtr = lPacket.data();
        std::memcpy( lPtr, &aSampler.mMinification, sizeof( eSamplerFilter ) );
        lPtr += sizeof( eSamplerFilter );
        std::memcpy( lPtr, &aSampler.mMagnification, sizeof( eSamplerFilter ) );
        lPtr += sizeof( eSamplerFilter );
        std::memcpy( lPtr, &aSampler.mMip, sizeof( eSamplerMipmap ) );
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

        return lPacket;
    }

    std::tuple<TextureData2D, TextureSampler2D> BinaryAsset::Retrieve( uint32_t aIndex )
    {
        auto lAssetIndex = mAssetIndex[aIndex];
        if( lAssetIndex.mType != eAssetType::KTX_TEXTURE_2D ) throw std::runtime_error( "Binary data type mismatch" );

        Seek( lAssetIndex.mByteStart );

        sTextureSamplingInfo lSamplerCreateInfo{};
        lSamplerCreateInfo.mMinification   = Read<eSamplerFilter>();
        lSamplerCreateInfo.mMagnification  = Read<eSamplerFilter>();
        lSamplerCreateInfo.mMip            = Read<eSamplerMipmap>();
        lSamplerCreateInfo.mWrapping       = Read<eSamplerWrapping>();
        lSamplerCreateInfo.mScaling[0]     = Read<float>();
        lSamplerCreateInfo.mScaling[1]     = Read<float>();
        lSamplerCreateInfo.mOffset[0]      = Read<float>();
        lSamplerCreateInfo.mOffset[1]      = Read<float>();
        lSamplerCreateInfo.mBorderColor[0] = Read<float>();
        lSamplerCreateInfo.mBorderColor[1] = Read<float>();
        lSamplerCreateInfo.mBorderColor[2] = Read<float>();
        lSamplerCreateInfo.mBorderColor[3] = Read<float>();

        uint32_t lKTXDataSize = lAssetIndex.mByteEnd - static_cast<uint32_t>( CurrentPosition() );

        auto lData = Read<char>( lKTXDataSize );

        TextureData2D    lTextureData( lData.data(), lData.size() );
        TextureSampler2D lSampler( lTextureData, lSamplerCreateInfo );

        return { lTextureData, lSampler };
    }

    std::vector<char> BinaryAsset::Package( std::vector<VertexData> const &aVertexData, std::vector<uint32_t> const &aIndexData )
    {
        uint32_t lHeaderSize = 2 * sizeof( uint32_t );
        uint32_t lPacketSize = aVertexData.size() * sizeof( VertexData ) + aIndexData.size() * sizeof( uint32_t ) + lHeaderSize;

        std::vector<char> lPacket( lPacketSize );

        uint32_t lVertexByteSize = aVertexData.size();
        uint32_t lIndexByteSize  = aIndexData.size();

        auto *lPtr = lPacket.data();
        std::memcpy( lPtr, &lVertexByteSize, sizeof( uint32_t ) );
        lPtr += sizeof( uint32_t );
        std::memcpy( lPtr, &lIndexByteSize, sizeof( uint32_t ) );
        lPtr += sizeof( uint32_t );
        std::memcpy( lPtr, aVertexData.data(), aVertexData.size() * sizeof( VertexData ) );
        lPtr += aVertexData.size() * sizeof( VertexData );
        std::memcpy( lPtr, aIndexData.data(), aIndexData.size() * sizeof( uint32_t ) );

        return lPacket;
    }

    void BinaryAsset::Retrieve( uint32_t aIndex, std::vector<VertexData> &aVertexData, std::vector<uint32_t> &aIndexData )
    {
        auto lAssetIndex = mAssetIndex[aIndex];
        if( lAssetIndex.mType != eAssetType::MESH_DATA ) throw std::runtime_error( "Binary data type mismatch" );

        Seek( lAssetIndex.mByteStart );

        auto lVertexBufferSize = Read<uint32_t>();
        auto lIndexBufferSize  = Read<uint32_t>();

        aVertexData = Read<VertexData>( lVertexBufferSize );
        aIndexData  = Read<uint32_t>( lIndexBufferSize );
    }

    std::vector<char> BinaryAsset::Package( sMaterial const &aMaterialData )
    {
        uint32_t lHeaderSize = sizeof( uint32_t );
        uint32_t lPacketSize = sizeof( sMaterial ) - sizeof( std::string ) + ( sizeof( uint32_t ) + aMaterialData.mName.size() );

        std::vector<char> lPacket( lPacketSize );
        auto             *lPtr          = lPacket.data();
        uint32_t          lNameByteSize = aMaterialData.mName.size();

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

        return lPacket;
    }

    void BinaryAsset::Retrieve( uint32_t aIndex, sMaterial &aMaterialData )
    {
        auto lAssetIndex = mAssetIndex[aIndex];
        if( lAssetIndex.mType != eAssetType::MATERIAL_DATA ) throw std::runtime_error( "Binary data type mismatch" );

        Seek( lAssetIndex.mByteStart );

        aMaterialData.mName              = Read<std::string>();
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

    std::vector<char> BinaryAsset::Package( sImportedAnimationSampler const &aMaterialData )
    {
        uint32_t lHeaderSize = sizeof( uint32_t ) + sizeof( uint8_t );
        uint32_t lPacketSize =
            lHeaderSize + aMaterialData.mInputs.size() * sizeof( float ) + aMaterialData.mOutputsVec4.size() * sizeof( math::vec4 );

        std::vector<char> lPacket( lPacketSize );
        auto             *lPtr = lPacket.data();

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

        return lPacket;
    }

    void BinaryAsset::Retrieve( uint32_t aIndex, sImportedAnimationSampler &aMaterialData )
    {
        auto lAssetIndex = mAssetIndex[aIndex];
        if( lAssetIndex.mType != eAssetType::ANIMATION_DATA ) throw std::runtime_error( "Binary data type mismatch" );

        Seek( lAssetIndex.mByteStart );

        aMaterialData.mInterpolation = Read<sImportedAnimationSampler::Interpolation>();

        auto lInputSize       = Read<uint32_t>();
        aMaterialData.mInputs = Read<float>( lInputSize );

        auto lOutputSize           = Read<uint32_t>();
        aMaterialData.mOutputsVec4 = Read<math::vec4>( lOutputSize );

        LTSE::Logging::Info("{}", aIndex);
    }

} // namespace LTSE::Core