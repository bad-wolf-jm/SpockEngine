#pragma once

#include "IDescriptorSetLayout.h"

namespace SE::Graphics
{
    struct sShaderData
    {
        eShaderStageTypeFlags mShaderType;

        path_t    mPath;
        string_t mEntryPoint;
    };

    uint32_t BufferDataTypeSize( eBufferDataType aType );

    struct sBufferLayoutElement
    {
        string_t     mName;
        eBufferDataType mType;
        uint32_t        mBinding;
        uint32_t        mLocation;
        size_t          mSize;
        size_t          mOffset;

        sBufferLayoutElement() = default;
        sBufferLayoutElement( const string_t &aName, eBufferDataType aType, uint32_t aBinding, uint32_t aLocation );
    };

    struct sDepthTesting
    {
        bool                   mDepthWriteEnable = false;
        bool                   mDepthTestEnable  = false;
        eDepthCompareOperation mDepthComparison  = eDepthCompareOperation::ALWAYS;
    };

    struct sBlending
    {
        bool mEnable = false;

        eBlendFactor    mSourceColorFactor   = eBlendFactor::ZERO;
        eBlendFactor    mDestColorFactor     = eBlendFactor::ZERO;
        eBlendOperation mColorBlendOperation = eBlendOperation::MAX;

        eBlendFactor    mSourceAlphaFactor   = eBlendFactor::ZERO;
        eBlendFactor    mDestAlphaFactor     = eBlendFactor::ZERO;
        eBlendOperation mAlphaBlendOperation = eBlendOperation::MAX;
    };

    struct sPushConstantRange
    {
        ShaderStageType mShaderStages = { eShaderStageTypeFlags::VERTEX };
        uint32_t        mOffset       = 0;
        uint32_t        mSize         = 0;
    };

    class IRenderContext;

    class IGraphicsPipeline
    {
      public:
        IGraphicsPipeline( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext, ePrimitiveTopology aTopology );
        ~IGraphicsPipeline() = default;

      public:
        virtual void Build() = 0;

      public:
        void SetShader( eShaderStageTypeFlags aShaderType, path_t aPath, string_t aEntryPoint );
        void SetCulling( eFaceCulling aCulling );
        void SetLineWidth( float aLineWidth );
        void SetDepthParameters( bool aDepthWriteEnable, bool aDepthTestEnable, eDepthCompareOperation aDepthComparison );
        void AddPushConstantRange( ShaderStageType aShaderStage, uint32_t aOffset, uint32_t aSize );

        void AddInput( string_t aName, eBufferDataType aType, uint32_t aBinding, uint32_t aLocation, bool aInstanced = false );

        template <typename _Ty>
        void AddPushConstantRange( ShaderStageType aShaderStage, uint32_t aOffset )
        {
            AddPushConstantRange( aShaderStage, aOffset, sizeof( _Ty ) );
        }

        void AddDescriptorSet( Ref<IDescriptorSetLayout> aDescriptorSet );

      protected:
        bool mOpaque = false;

        Ref<IGraphicContext> mGraphicContext = nullptr;
        Ref<IRenderContext>  mRenderContext  = nullptr;

        ePrimitiveTopology mTopology    = ePrimitiveTopology::TRIANGLES;
        eFaceCulling       mCulling     = eFaceCulling::BACK;
        uint8_t            mSampleCount = 1;
        float              mLineWidth   = 1.0f;

        bool                   mDepthWriteEnable = false;
        bool                   mDepthTestEnable  = false;
        eDepthCompareOperation mDepthComparison  = eDepthCompareOperation::ALWAYS;

        std::vector<sShaderData>               mShaderStages         = {};
        std::vector<sBufferLayoutElement>      mInputLayout          = {};
        std::vector<sBufferLayoutElement>      mInstancedInputLayout = {};
        std::vector<sPushConstantRange>        mPushConstants        = {};
        std::vector<Ref<IDescriptorSetLayout>> mDescriptorSets       = {};
    };
} // namespace SE::Graphics