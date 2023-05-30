#include "VkShaderProgram.h"

#include "Core/Core.h"
#include "Core/Memory.h"
#include "Graphics/Vulkan/VkCoreMacros.h"

#include "Core/Logging.h"
#include "Shader/Compiler.h"

namespace SE::Graphics
{
    VkShaderProgram::VkShaderProgram( Ref<IGraphicContext> aGraphicContext, eShaderStageTypeFlags aShaderType, int aVersion )
        : IShaderProgram( aGraphicContext, aShaderType, aVersion )
    {
    }

    void VkShaderProgram::Compile() { SE::Graphics::Compile( mShaderType, Program(), mCompiledByteCode ); }
} // namespace SE::Graphics
