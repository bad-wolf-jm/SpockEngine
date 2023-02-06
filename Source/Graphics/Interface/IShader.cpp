#include "IShader.h"

namespace SE::Graphics
{

    IShader::IShader()                     = default;
    IShader::IShader( IGraphicResource & ) = default;

    IShader::IShader( Ref<IGraphicContext> aGraphicContext, eShaderType aType )
        : mGraphicContext{ aGraphicContext }
        , mType{ aType }
    {
    }
} // namespace SE::Graphics