#include "IShaderProgram.h"

#include <fmt/core.h>
#include <fstream>
#include <sstream>

namespace SE::Graphics
{
    IShaderProgram::IShaderProgram( Ref<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion,
                                    std::string const &aName, fs::path const &aCacheRoot )
        : mGraphicContext{ aGraphicContext }
        , mVersion{ aVersion }
        , mShaderType{ aShaderType }
        , mName{ aName }
        , mCacheRoot{ aCacheRoot }
    {
    }

    IShaderProgram::IShaderProgram( Ref<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion,
                                    std::string const &aName )
        : IShaderProgram( aGraphicContext, aShaderType, aVersion, aName, "" )
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

    void IShaderProgram::Compile()
    {
        mCacheFileName = fmt::format( "shader_{}_{}.spv", mName, Hash() );

        if( fs::exists( mCacheRoot / mCacheFileName ) )
        {
            auto lShaderCode  = ReadFile( mCacheRoot / mCacheFileName );
            mCompiledByteCode = std::vector<uint32_t>( lShaderCode.size() / sizeof( uint32_t ) );

            std::memcpy( mCompiledByteCode.data(), lShaderCode.data(), lShaderCode.size() );
        }
        else
        {
            DoCompile();

            std::ofstream lFileObject( mCacheRoot / mCacheFileName, std::ios::out | std::ios::binary );
            lFileObject.write( (char *)mCompiledByteCode.data(), mCompiledByteCode.size() * sizeof( uint32_t ) );
            lFileObject.close();
        }

        BuildProgram();
    }

} // namespace SE::Graphics
