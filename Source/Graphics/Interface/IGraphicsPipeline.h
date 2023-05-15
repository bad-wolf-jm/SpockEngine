#pragma once

#include "IDescriptorSetLayout.h"

namespace SE::Graphics
{
    struct sShaderData
    {
        eShaderStageTypeFlags mShaderType;

        fs::path    mPath;
        std::string mEntryPoint;
    };

    enum class eBufferDataType : uint32_t
    {
        UINT8  = 0,
        UINT16 = 1,
        UINT32 = 2,
        INT8   = 3,
        INT16  = 4,
        INT32  = 5,
        FLOAT  = 6,
        COLOR  = 7,
        VEC2   = 8,
        VEC3   = 9,
        VEC4   = 10,
        IVEC2  = 11,
        IVEC3  = 12,
        IVEC4  = 13,
        UVEC2  = 14,
        UVEC3  = 15,
        UVEC4  = 16
    };

    uint32_t BufferDataTypeSize( eBufferDataType aType );

    struct sBufferLayoutElement
    {
        std::string     mName;
        eBufferDataType mType;
        uint32_t        mBinding;
        uint32_t        mLocation;
        size_t          mSize;
        size_t          mOffset;

        sBufferLayoutElement() = default;
        sBufferLayoutElement( const std::string &aName, eBufferDataType aType, uint32_t aBinding, uint32_t aLocation );
    };

    enum class eDepthCompareOperation : uint32_t
    {
        NEVER            = 0,
        LESS             = 1,
        EQUAL            = 2,
        LESS_OR_EQUAL    = 3,
        GREATER          = 4,
        NOT_EQUAL        = 5,
        GREATER_OR_EQUAL = 6,
        ALWAYS           = 7
    };

    struct sDepthTesting
    {
        bool                   mDepthWriteEnable = false;
        bool                   mDepthTestEnable  = false;
        eDepthCompareOperation mDepthComparison  = eDepthCompareOperation::ALWAYS;
    };

    enum class eBlendOperation : uint32_t
    {
        ADD              = 0,
        SUBTRACT         = 1,
        REVERSE_SUBTRACT = 2,
        MIN              = 3,
        MAX              = 4
    };

    enum class eBlendFactor : uint32_t
    {
        ZERO                     = 0,
        ONE                      = 1,
        SRC_COLOR                = 2,
        ONE_MINUS_SRC_COLOR      = 3,
        DST_COLOR                = 4,
        ONE_MINUS_DST_COLOR      = 5,
        SRC_ALPHA                = 6,
        ONE_MINUS_SRC_ALPHA      = 7,
        DST_ALPHA                = 8,
        ONE_MINUS_DST_ALPHA      = 9,
        CONSTANT_COLOR           = 10,
        ONE_MINUS_CONSTANT_COLOR = 11,
        CONSTANT_ALPHA           = 12,
        ONE_MINUS_CONSTANT_ALPHA = 13,
        SRC_ALPHA_SATURATE       = 14,
        SRC1_COLOR               = 15,
        ONE_MINUS_SRC1_COLOR     = 16,
        SRC1_ALPHA               = 17,
        ONE_MINUS_SRC1_ALPHA     = 18
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

    enum class ePrimitiveTopology : uint32_t
    {
        POINTS    = 0,
        TRIANGLES = 1,
        LINES     = 2
    };

    enum class eFaceCulling : uint32_t
    {
        NONE           = 0,
        FRONT          = 1,
        BACK           = 2,
        FRONT_AND_BACK = 3
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
        void SetShader( eShaderStageTypeFlags aShaderType, fs::path aPath, std::string aEntryPoint );
        void SetCulling( eFaceCulling aCulling );
        void SetLineWidth( float aLineWidth );
        void SetDepthParameters( bool aDepthWriteEnable, bool aDepthTestEnable, eDepthCompareOperation aDepthComparison );
        void AddPushConstantRange( ShaderStageType aShaderStage, uint32_t aOffset, uint32_t aSize );

        void AddInput( std::string aName, eBufferDataType aType, uint32_t aBinding, uint32_t aLocation, bool aInstanced = false );

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