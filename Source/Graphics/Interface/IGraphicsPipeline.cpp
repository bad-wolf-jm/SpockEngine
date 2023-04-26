#include "IGraphicsPipeline.h"

namespace SE::Graphics
{
    IGraphicsPipeline::IGraphicsPipeline( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext, ePrimitiveTopology aTopology )
        : mGraphicContext{ aGraphicContext }
        , mRenderContext{ aRenderContext }
        , mTopology{aTopology}
    {
    }

    void IGraphicsPipeline::SetCulling( eFaceCulling aCulling ) { mCulling = aCulling; }

    void IGraphicsPipeline::SetLineWidth( float aLineWidth ) { mLineWidth = aLineWidth; }

    void IGraphicsPipeline::SetDepthParameters( bool aDepthWriteEnable, bool aDepthTestEnable,
                                                eDepthCompareOperation aDepthComparison )
    {
        mDepthWriteEnable = aDepthWriteEnable;
        mDepthTestEnable  = aDepthTestEnable;
        mDepthComparison  = aDepthComparison;
    }

    void IGraphicsPipeline::AddPushConstantRange( ShaderStageType aShaderStage, uint32_t aOffset, uint32_t aSize )
    {
        mPushConstants.push_back( sPushConstantRange{ aShaderStage, aOffset, aSize } );
    }

    void IGraphicsPipeline::SetShader( eShaderStageTypeFlags aShaderType, fs::path aPath, std::string aEntryPoint )
    {
        mShaderStages.push_back( sShaderData{ aShaderType, aPath, aEntryPoint } );
    }

} // namespace SE::Graphics