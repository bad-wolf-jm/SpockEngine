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

    void IGraphicsPipeline::SetShader( eShaderStageTypeFlags aShaderType, path_t aPath, string_t aEntryPoint )
    {
        mShaderStages.push_back( sShaderData{ aShaderType, aPath, aEntryPoint } );
    }

    void IGraphicsPipeline::AddDescriptorSet( Ref<IDescriptorSetLayout> aDescriptorSet ) { mDescriptorSets.push_back( aDescriptorSet ); }

    void IGraphicsPipeline::AddInput( string_t aName, eBufferDataType aType, uint32_t aBinding, uint32_t aLocation,
                                      bool aInstanced )
    {
        auto &lInputDescription = ( aInstanced ? mInstancedInputLayout.emplace_back() : mInputLayout.emplace_back() );

        lInputDescription.mName     = aName;
        lInputDescription.mType     = aType;
        lInputDescription.mBinding  = aBinding;
        lInputDescription.mLocation = aLocation;
        lInputDescription.mSize     = BufferDataTypeSize( aType );
    }

    uint32_t BufferDataTypeSize( eBufferDataType aType )
    {
        // clang-format off
        switch( aType )
        {
        case eBufferDataType::UINT8:  return 1 * sizeof( uint8_t  );
        case eBufferDataType::UINT16: return 1 * sizeof( uint16_t );
        case eBufferDataType::UINT32: return 1 * sizeof( uint32_t );
        case eBufferDataType::INT8:   return 1 * sizeof( int8_t   );
        case eBufferDataType::INT16:  return 1 * sizeof( int16_t  );
        case eBufferDataType::INT32:  return 1 * sizeof( int32_t  );
        case eBufferDataType::FLOAT:  return 1 * sizeof( float    );
        case eBufferDataType::COLOR:  return 4 * sizeof( uint8_t  );
        case eBufferDataType::VEC2:   return 2 * sizeof( float    );
        case eBufferDataType::VEC3:   return 3 * sizeof( float    );
        case eBufferDataType::VEC4:   return 4 * sizeof( float    );
        case eBufferDataType::IVEC2:  return 2 * sizeof( int32_t  );
        case eBufferDataType::IVEC3:  return 3 * sizeof( int32_t  );
        case eBufferDataType::IVEC4:  return 4 * sizeof( int32_t  );
        case eBufferDataType::UVEC2:  return 2 * sizeof( uint32_t );
        case eBufferDataType::UVEC3:  return 3 * sizeof( uint32_t );
        case eBufferDataType::UVEC4:  return 4 * sizeof( uint32_t );
        }
        // clang-format on
        return 0;
    };


    sBufferLayoutElement::sBufferLayoutElement( const string_t &aName, eBufferDataType aType, uint32_t aBinding,
                                                uint32_t aLocation )
        : mName( aName )
        , mType( aType )
        , mBinding( aBinding )
        , mLocation( aLocation )
        , mSize( BufferDataTypeSize( aType ) )
        , mOffset( 0 )
    {
    }


} // namespace SE::Graphics