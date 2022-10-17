#include "AssetFile.h"

namespace LTSE::Core
{
    static constexpr uint8_t sFileMagic[] = { '@', '%', 'S', 'P', 'O', 'C', 'K', 'E', 'N', 'G', 'I', 'N', 'E', '_', 'S', 'C',
        'E', 'N', 'E', '_', 'A', 'S', 'S', 'E', 'T', '@', '%' };

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

    std::vector<char> BinaryAsset::Package( TextureData2D const &aData, TextureSampler2D const &aSampler )
    {

        uint32_t lHeaderSize = 0;
        lHeaderSize +=
            sizeof( eSamplerFilter ) + sizeof( eSamplerFilter ) + sizeof( eSamplerMipmap ) + sizeof( eSamplerWrapping );
        lHeaderSize += 2 * sizeof( float );
        lHeaderSize += 2 * sizeof( float );
        lHeaderSize += 4 * sizeof( float );

        auto     lKTXData    = aData.Serialize();
        uint32_t lPacketSize = lKTXData.size() + lHeaderSize;

        std::vector<char> lPacket( lPacketSize );

        auto *lPtr = lPacket.data();
        std::memcpy( lPtr, &aSampler.mSamplingSpec.mMinification, sizeof( eSamplerFilter ) );
        lPtr += sizeof( eSamplerFilter );
        std::memcpy( lPtr, &aSampler.mSamplingSpec.mMagnification, sizeof( eSamplerFilter ) );
        lPtr += sizeof( eSamplerFilter );
        std::memcpy( lPtr, &aSampler.mSamplingSpec.mMip, sizeof( eSamplerMipmap ) );
        lPtr += sizeof( eSamplerMipmap );
        std::memcpy( lPtr, &aSampler.mSamplingSpec.mWrapping, sizeof( eSamplerWrapping ) );
        lPtr += sizeof( eSamplerWrapping );

        std::memcpy( lPtr, &aSampler.mSamplingSpec.mScaling[0], sizeof( float ) );
        lPtr += sizeof( float );
        std::memcpy( lPtr, &aSampler.mSamplingSpec.mScaling[1], sizeof( float ) );
        lPtr += sizeof( float );

        std::memcpy( lPtr, &aSampler.mSamplingSpec.mOffset[0], sizeof( float ) );
        lPtr += sizeof( float );
        std::memcpy( lPtr, &aSampler.mSamplingSpec.mOffset[1], sizeof( float ) );
        lPtr += sizeof( float );

        std::memcpy( lPtr, &aSampler.mSamplingSpec.mBorderColor[0], sizeof( float ) );
        lPtr += sizeof( float );
        std::memcpy( lPtr, &aSampler.mSamplingSpec.mBorderColor[1], sizeof( float ) );
        lPtr += sizeof( float );
        std::memcpy( lPtr, &aSampler.mSamplingSpec.mBorderColor[2], sizeof( float ) );
        lPtr += sizeof( float );
        std::memcpy( lPtr, &aSampler.mSamplingSpec.mBorderColor[3], sizeof( float ) );
        lPtr += sizeof( float );

        std::memcpy( lPtr, lKTXData.data(), lKTXData.size() );

        return lPacket;
    }

    std::vector<char> BinaryAsset::Package(
        std::vector<VertexData> const &aVertexData, std::vector<uint32_t> const &aIndexData )
    {
        return {};
    }

    std::vector<char> BinaryAsset::Package( sMaterial const &aMaterialData ) { return {}; }

    std::vector<char> BinaryAsset::Package( sImportedAnimationSampler const &aMaterialData ) { return {}; }

    void BinaryAsset::Retrieve( uint32_t aIndex, Core::TextureData2D &aData, Core::TextureSampler2D &aSampler )
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

        aData    = TextureData2D( lData.data(), lData.size() );
        aSampler = TextureSampler2D( aData, lSamplerCreateInfo );
    }

    void BinaryAsset::Retrieve( uint32_t aIndex, std::vector<VertexData> &aVertexData, std::vector<uint32_t> &aIndexData )
    {
        auto lAssetIndex = mAssetIndex[aIndex];
        if( lAssetIndex.mType != eAssetType::MESH_DATA ) throw std::runtime_error( "Binary data type mismatch" );

        Seek( lAssetIndex.mByteStart );
    }

    void BinaryAsset::Retrieve( uint32_t aIndex, sMaterial &aMaterialData )
    {
        auto lAssetIndex = mAssetIndex[aIndex];
        if( lAssetIndex.mType != eAssetType::MATERIAL_DATA ) throw std::runtime_error( "Binary data type mismatch" );

        Seek( lAssetIndex.mByteStart );
    }

    void BinaryAsset::Retrieve( uint32_t aIndex, sImportedAnimationSampler &aMaterialData )
    {
        auto lAssetIndex = mAssetIndex[aIndex];
        if( lAssetIndex.mType != eAssetType::ANIMATION_DATA ) throw std::runtime_error( "Binary data type mismatch" );

        Seek( lAssetIndex.mByteStart );
    }

} // namespace LTSE::Core