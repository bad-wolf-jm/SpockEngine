#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"
#include "Core/Textures/TextureData.h"

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>

#include "Scene/Serialize/AssetFile.h"
#include "Scene/Serialize/FileIO.h"

using namespace LTSE::Core;

sImageData LoadBinData( fs::path const &aPath )
{
    constexpr size_t lComponentCount = 4;

    sImageData lImageData{};
    int32_t    lActualComponentCount = 0;
    size_t     lChannelSize          = 0;

    std::ifstream lInputFile;
    lInputFile.open( aPath, std::ios::in | std::ios::binary );
    uint32_t lHeight, lWidth;
    lInputFile.seekg( 0, std::ios::beg );
    lInputFile.read( (char *)&lHeight, sizeof( uint32_t ) );
    lInputFile.read( (char *)&lWidth, sizeof( uint32_t ) );
    lInputFile.seekg( 2 * sizeof( uint32_t ), std::ios::beg );

    float *lValue = (float *)malloc( lHeight * lWidth * sizeof( float ) );
    lInputFile.read( (char *)lValue, lHeight * lWidth * sizeof( float ) );

    lImageData.mWidth     = static_cast<size_t>( lWidth );
    lImageData.mHeight    = static_cast<size_t>( lHeight );
    lImageData.mFormat    = eColorFormat::R32_FLOAT;
    lImageData.mByteSize  = lHeight * lWidth * sizeof( float );
    lImageData.mPixelData = (uint8_t *)lValue;

    return lImageData;
}

std::vector<char> ConvertToKTX( fs::path const &aPath )
{
    sImageData lBinaryData{};
    if( aPath.extension().string() == ".bin" )
    {
        lBinaryData = LoadBinData( aPath );

        LTSE::Core::TextureData::sCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mMipLevels = 1;
        LTSE::Core::TextureData2D lTexture( lTextureCreateInfo, lBinaryData );

        return lTexture.Serialize();
    }
    else if( aPath.extension().string() == ".ktx" )
    {
        LTSE::Core::TextureData::sCreateInfo lTextureCreateInfo{};
        LTSE::Core::TextureData2D            lTexture( lTextureCreateInfo, aPath );

        return lTexture.Serialize();
    }
    else
    {
        lBinaryData = LoadImageData( aPath );
        LTSE::Core::TextureData::sCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mMipLevels = 1;
        LTSE::Core::TextureData2D lTexture( lTextureCreateInfo, lBinaryData );

        return lTexture.Serialize();
    }
}

eSamplerFilter GetFilter( std::string const &aKey )
{
    std::unordered_map<std::string, eSamplerFilter> lValues = {
        { "nearest", eSamplerFilter::NEAREST }, { "linear", eSamplerFilter::LINEAR } };
    return lValues[aKey];
}

eSamplerMipmap GetMipFilter( std::string const &aKey )
{
    std::unordered_map<std::string, eSamplerMipmap> lValues = {
        { "nearest", eSamplerMipmap::NEAREST }, { "linear", eSamplerMipmap::LINEAR } };
    return lValues[aKey];
}

eSamplerWrapping GetWrapping( std::string const &aKey )
{
    std::unordered_map<std::string, eSamplerWrapping> lValues = { { "repeat", eSamplerWrapping::REPEAT },
        { "mirrored_repeat", eSamplerWrapping::MIRRORED_REPEAT }, { "clamp_to_edge", eSamplerWrapping::CLAMP_TO_EDGE },
        { "clamp_to_border", eSamplerWrapping::CLAMP_TO_BORDER },
        { "mirror_clamnp_to_border", eSamplerWrapping::MIRROR_CLAMP_TO_BORDER } };
    return lValues[aKey];
}

std::vector<char> MakePacket( sTextureSamplingInfo aSampling, std::vector<char> aKTXData )
{
    uint32_t lHeaderSize = 0;
    lHeaderSize +=
        sizeof( eSamplerFilter ) + sizeof( eSamplerFilter ) + sizeof( eSamplerMipmap ) + sizeof( eSamplerWrapping );
    lHeaderSize += 2 * sizeof( float );
    lHeaderSize += 2 * sizeof( float );
    lHeaderSize += 4 * sizeof( float );

    uint32_t lPacketSize = aKTXData.size() + lHeaderSize;

    std::vector<char> lPacket( lPacketSize );
    auto             *lPtr = lPacket.data();
    std::memcpy( lPtr, &aSampling.mMinification, sizeof( eSamplerFilter ) );
    lPtr += sizeof( eSamplerFilter );
    std::memcpy( lPtr, &aSampling.mMagnification, sizeof( eSamplerFilter ) );
    lPtr += sizeof( eSamplerFilter );
    std::memcpy( lPtr, &aSampling.mMip, sizeof( eSamplerMipmap ) );
    lPtr += sizeof( eSamplerMipmap );
    std::memcpy( lPtr, &aSampling.mWrapping, sizeof( eSamplerWrapping ) );
    lPtr += sizeof( eSamplerWrapping );

    std::memcpy( lPtr, &aSampling.mScaling[0], sizeof( float ) );
    lPtr += sizeof( float );
    std::memcpy( lPtr, &aSampling.mScaling[1], sizeof( float ) );
    lPtr += sizeof( float );

    std::memcpy( lPtr, &aSampling.mOffset[0], sizeof( float ) );
    lPtr += sizeof( float );
    std::memcpy( lPtr, &aSampling.mOffset[1], sizeof( float ) );
    lPtr += sizeof( float );

    std::memcpy( lPtr, &aSampling.mBorderColor[0], sizeof( float ) );
    lPtr += sizeof( float );
    std::memcpy( lPtr, &aSampling.mBorderColor[1], sizeof( float ) );
    lPtr += sizeof( float );
    std::memcpy( lPtr, &aSampling.mBorderColor[2], sizeof( float ) );
    lPtr += sizeof( float );
    std::memcpy( lPtr, &aSampling.mBorderColor[3], sizeof( float ) );
    lPtr += sizeof( float );

    std::memcpy( lPtr, aKTXData.data(), aKTXData.size() );

    return lPacket;
}

int main( int argc, char **argv )
{
    argparse::ArgumentParser lProgramArguments( "bin2ktx" );

    lProgramArguments.add_argument( "-i", "--input" ).help( "Specify input file" );
    lProgramArguments.add_argument( "-o", "--output" ).help( "Specify output file" );

    try
    {
        lProgramArguments.parse_args( argc, argv );
    }
    catch( const std::runtime_error &err )
    {
        std::cerr << err.what() << std::endl;
        std::cerr << lProgramArguments;
        std::exit( 1 );
    }

    auto lInput  = fs::path( lProgramArguments.get<std::string>( "--input" ) );
    auto lOutput = fs::path( lProgramArguments.get<std::string>( "--output" ) );

    LTSE::Logging::Info( "Input: {}", lInput.string() );
    LTSE::Logging::Info( "Output: {}", lOutput.string() );

    if( !fs::exists( lInput ) )
    {
        LTSE::Logging::Info( "Input file '{}' does not exist", lInput.string() );
        std::exit( 1 );
    }

    auto *lMagic       = BinaryAsset::GetMagic();
    auto  lMagicLength = BinaryAsset::GetMagicLength();

    auto lOutFile = std::ofstream( lOutput.string(), std::ofstream::binary );
    lOutFile.write( (const char *)lMagic, lMagicLength );

    std::vector<sAssetIndex>       lAssetIndex{};
    std::vector<std::vector<char>> lPackets{};

    ConfigurationReader lConfigFile( lInput );
    ConfigurationNode   lRootNode = lConfigFile.GetRoot();
    lRootNode["assets"].ForEach(
        [&]( ConfigurationNode &aValue )
        {
            auto lRelativeFilePath = aValue["file"].As<std::string>( "" );
            auto lAssetType        = aValue["type"].As<std::string>( "" );

            auto lAbsoluteFilePath = lInput.parent_path() / lRelativeFilePath;

            if( lAssetType == "texture_2d" )
            {
                auto                 lProperties = aValue["properties"];
                sTextureSamplingInfo lSamplingInfo{};
                lSamplingInfo.mMinification  = GetFilter( lProperties["filttering.min"].As<std::string>( "linear" ) );
                lSamplingInfo.mMagnification = GetFilter( lProperties["filttering.max"].As<std::string>( "linear" ) );
                lSamplingInfo.mMip           = GetMipFilter( lProperties["mipmap"].As<std::string>( "linear" ) );
                lSamplingInfo.mWrapping      = GetWrapping( lProperties["wrapping"].As<std::string>( "clamp_to_border" ) );

                auto lOffset          = lProperties["offset"].Vec( { "x", "y" }, math::vec2{ 0.0f, 0.0f } );
                lSamplingInfo.mOffset = { lOffset.x, lOffset.y };

                auto lScaling          = lProperties["scaling"].Vec( { "x", "y" }, math::vec2{ 1.0f, 1.0f } );
                lSamplingInfo.mScaling = { lScaling.x, lScaling.y };

                auto lBorderColor =
                    lProperties["border_color"].Vec( { "x", "y", "z", "w" }, math::vec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
                lSamplingInfo.mBorderColor = { lBorderColor.x, lBorderColor.y, lBorderColor.z, lBorderColor.w };

                sAssetIndex lAssetIndexEntry{};
                lAssetIndexEntry.mType      = eAssetType::KTX_TEXTURE_2D;
                lAssetIndexEntry.mByteStart = 0;
                lAssetIndexEntry.mByteEnd   = 1;
                lAssetIndex.push_back( lAssetIndexEntry );

                auto lKTXData       = ConvertToKTX( lAbsoluteFilePath );
                auto lTexturePacket = MakePacket( lSamplingInfo, lKTXData );
                lPackets.push_back( lTexturePacket );

                LTSE::Logging::Info( "{} - {}", lAbsoluteFilePath.string(), lAssetType );
            }
        } );
    uint32_t lAssetCount = static_cast<uint32_t>( lAssetIndex.size() );
    lOutFile.write( (const char *)&lAssetCount, sizeof( uint32_t ) );

    uint32_t lCurrentByte = BinaryAsset::GetMagicLength() + sizeof( uint32_t ) + lAssetIndex.size() * sizeof( sAssetIndex );
    for( uint32_t i = 0; i < lAssetCount; i++ )
    {
        lAssetIndex[i].mByteStart = lCurrentByte;
        lCurrentByte = lAssetIndex[i].mByteEnd = lAssetIndex[i].mByteStart + static_cast<uint32_t>( lPackets[i].size() );
        lCurrentByte;
    }

    lOutFile.write( (const char *)lAssetIndex.data(), lAssetIndex.size() * sizeof( sAssetIndex ) );

    for( auto &lPacket : lPackets ) lOutFile.write( (const char *)lPacket.data(), lPacket.size() );

    return 0;
}
