#include "IShaderProgram.h"

#include <fmt/core.h>
#include <sstream>
#include <fstream>

namespace SE::Graphics
{
    IShaderProgram::IShaderProgram( Ref<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion )
        : mGraphicContext{ aGraphicContext }
        , mVersion{ aVersion }
        , mShaderType{aShaderType}
    {
    }

    void IShaderProgram::AddCode( std::string const &aCode ) { mCodeBlocks.push_back( aCode ); }

    static std::vector<char> ReadFile( const fs::path &filename )
    {
        std::ifstream lFileObject( filename, std::ios::ate | std::ios::binary );

        if( !lFileObject.is_open() ) throw std::runtime_error( "failed to open file!" );

        size_t            lFileSize = (size_t)lFileObject.tellg();
        std::vector<char> lBuffer( lFileSize );

        lFileObject.seekg( 0 );
        lFileObject.read( lBuffer.data(), lFileSize );
        lFileObject.close();

        return lBuffer;
    }

    void IShaderProgram::AddFile( fs::path const &aPath )
    {
        auto lContent = ReadFile( aPath );
        mCodeBlocks.emplace_back( lContent.begin(), lContent.end() );
    }

    std::string IShaderProgram::Program()
    {
        std::ostringstream lOutput;

        lOutput << fmt::format( "#version {}\n", mVersion );
        lOutput << "\n";

        for( auto const &lProgramFragment : mCodeBlocks ) lOutput << lProgramFragment;

        return lOutput.str();
    }

    std::string IShaderProgram::Hash()
    {
        std::stringstream stream;
        stream << std::hex << std::hash<std::string>{}( Program() );

        return stream.str();
    }

} // namespace SE::Graphics
