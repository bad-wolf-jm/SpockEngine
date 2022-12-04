#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"
#include "Core/CUDA/Texture/TextureData.h"

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>

using namespace SE::Core;

sImageData LoadBinData( fs::path &aPath )
{
    constexpr size_t lComponentCount = 4;

    sImageData lImageData{};
    int32_t lActualComponentCount = 0;
    size_t lChannelSize           = 0;

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

    SE::Logging::Info( "Input: {}", lInput.string() );
    SE::Logging::Info( "Output: {}", lOutput.string() );

    if( !fs::exists( lInput ) )
    {
        SE::Logging::Info( "Input file '{}' does not exist", lInput.string() );
        std::exit( 1 );
    }

    sImageData lBinaryData{};
    if( lInput.extension().string() == ".bin" )
    {
        lBinaryData = LoadBinData( lInput );
    }
    else
    {
        lBinaryData = LoadImageData( lInput );
    }

    std::cout << "============ IMAGE DATA ================" << std::endl;
    std::cout << "Width: " << lBinaryData.mWidth << std::endl;
    std::cout << "Height: " << lBinaryData.mHeight << std::endl;
    std::cout << "========================================" << std::endl;

    TextureData::sCreateInfo lTextureCreateInfo{};
    lTextureCreateInfo.mMipLevels = 1;
    TextureData2D lTexture( lTextureCreateInfo, lBinaryData );

    lTexture.SaveTo(lOutput);

    return 0;
}
