
#include "AssetImporter.h"

#include <iostream>

#include "Core/Logging.h"

namespace fs = std::filesystem;

#include <argparse/argparse.hpp>
#include "Developer/Scene/Importer/glTFImporter.h"

int main( int argc, char **argv )
{

    auto lImporter = GlTFImporter("C:\\work\\assets\\glTF-Sample-Models\\2.0\\Sponza\\glTF\\Sponza.gltf");
    // auto lImporter = GlTFImporter("C:\\work\\assets\\glTF-Sample-Models\\2.0\\BrainStem\\glTF\\BrainStem.gltf");
    // auto lImporter = GlTFImporter("C:\\work\\assets\\glTF-Sample-Models\\2.0\\BrainStem\\glTF-Binary\\BrainStem.glb");
    // auto lImporter = GlTFImporter("C:\\work\\assets\\Sponza\\Sponza.glb");

    // argparse::ArgumentParser lProgramArguments( "import_scene" );
    // lProgramArguments.add_argument( "-i", "--input" ).help( "Specify input file" );
    // lProgramArguments.add_argument( "-o", "--output" ).help( "Specify output folder" );

    // try
    // {
    //     lProgramArguments.parse_args( argc, argv );
    // }
    // catch( const std::runtime_error &err )
    // {
    //     std::cerr << err.what() << std::endl;
    //     std::cerr << lProgramArguments;
    //     std::exit( 1 );
    // }

    // auto lInput  = fs::path( lProgramArguments.get<std::string>( "--input" ) );
    // auto lOutput = fs::path( lProgramArguments.get<std::string>( "--output" ) );

    // LTSE::Logging::Info( "Input: {}", lInput.string() );
    // LTSE::Logging::Info( "Output: {}", lOutput.string() );

    // if( !fs::exists( lInput ) )
    // {
    //     LTSE::Logging::Info( "Input file '{}' does not exist", lInput.string() );
    // }

    // try
    // {
    //     ReadAssetFile(lInput, lOutput);
    // }
    // catch( std::exception e )
    // {
    //     std::cerr << "HELLO ERROR!!" << e.what();
    // }
}
