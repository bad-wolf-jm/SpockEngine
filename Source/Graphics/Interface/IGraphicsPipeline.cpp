#include "IGraphicsPipeline.h"

namespace SE::Graphics
{
    IGraphicsPipeline::IGraphicsPipeline( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext,
                                          ePrimitiveTopology aTopology )
        : mGraphicContext{ aGraphicContext }
        , mRenderContext{ aRenderContext }
        , mTopology{ aTopology }
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

    sDescriptorSet &IGraphicsPipeline::AddDescriptorSet( bool aUnbounded )
    {

        auto &lDescriptorSet = mDescriptorLayout.emplace_back();

        lDescriptorSet.mIsUnbounded = aUnbounded;

        return lDescriptorSet;
    }

    void sDescriptorSet::Add( uint32_t aBindingIndex, eDescriptorType aType, ShaderStageType aShaderStages )
    {
        mDescriptors.push_back( sDescriptorBindingInfo{ aBindingIndex, aType, aShaderStages } );
    }

    void IGraphicsPipeline::AddInput( std::string aName, eBufferDataType aType, uint32_t aBinding, uint32_t aLocation,
                                      bool aInstanced )
    {
        auto &lInputDescription = ( aInstanced ? mInstancedInputLayout.emplace_back() : mInputLayout.emplace_back() );

        lInputDescription.mName     = aName;
        lInputDescription.mType     = aType;
        lInputDescription.mBinding  = aBinding;
        lInputDescription.mLocation = aLocation;
        lInputDescription.mSize     = BufferDataTypeSize( aType );
    }

} // namespace SE::Graphics