#include "VkShaderProgram.h"

#include "Core/Core.h"
#include "Core/Memory.h"

#include "VkCoreMacros.h"

#include "Core/Logging.h"
#include "Shader/Compiler.h"

namespace SE::Graphics
{
    VkShaderProgram::VkShaderProgram( ref_t<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion,
                                      string_t const &aName, fs::path const &aCacheRoot )
        : IShaderProgram( aGraphicContext, aShaderType, aVersion, aName, aCacheRoot )
    {
    }

    void VkShaderProgram::DoCompile()
    {
        SE::Graphics::Compile( mShaderType, Program(), mCompiledByteCode );
    }

    void VkShaderProgram::BuildProgram()
    {
        auto const &lCachePath = ( mCacheRoot / mCacheFileName ).string();

        mShaderModule = New<ShaderModule>( Cast<VkGraphicContext>( mGraphicContext ), lCachePath, mShaderType );
    }
} // namespace SE::Graphics
