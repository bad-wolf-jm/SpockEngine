#include "AssetFile.h"

namespace LTSE::Core
{
    static constexpr uint8_t sFileMagic[] = { '@', '%', 'L', 'E', 'D', 'D', 'A', 'R', '_', 'E', 'C', 'H', 'O', '_', 'A', 'S', 'S', 'E', 'T', '@', '%' };

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
        if( memcmp( lMagic.data(), sFileMagic, sizeof( sFileMagic ) != 0) )
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
        if( lAssetIndex.mType != eAssetType::KTX_TEXTURE_2D )
            throw std::runtime_error( "Binary data type mismatch" );

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

        TextureData2D lTextureData( lData.data(), lData.size() );
        TextureSampler2D lSampler( lTextureData, lSamplerCreateInfo );

        return { lTextureData, lSampler };
    }

} // namespace LTSE::Core