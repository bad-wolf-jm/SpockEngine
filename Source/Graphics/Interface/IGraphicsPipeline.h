#pragma once

#include "Graphics/Vulkan/VkPipeline.h"

namespace SE::Graphics
{
    struct sShaderData
    {
        eShaderStageTypeFlags mShaderType;

        fs::path    mPath;
        std::string mEntryPoint;
    };

    struct sDescriptorBindingInfo
    {
        uint32_t        mBindingIndex = 0;
        eDescriptorType mType         = eDescriptorType::UNIFORM_BUFFER;
        ShaderStageType mShaderStages = {};
    };

    struct sDescriptorSet
    {
        bool mIsUnbounded = false;
        std::vector<sDescriptorBindingInfo> mDescriptors = {};

        void Add( uint32_t aBindingIndex, eDescriptorType aType, ShaderStageType aShaderStages );
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

        sDescriptorSet &AddDescriptorSet(bool aUnbounded = false);

      protected:
        bool mOpaque = false;

        Ref<IGraphicContext> mGraphicContext = nullptr;
        Ref<IRenderContext>  mRenderContext  = nullptr;

        ePrimitiveTopology mTopology             = ePrimitiveTopology::TRIANGLES;
        eFaceCulling       mCulling              = eFaceCulling::BACK;
        uint8_t            mSampleCount          = 1;
        float              mLineWidth            = 1.0f;

        bool                   mDepthWriteEnable = false;
        bool                   mDepthTestEnable  = false;
        eDepthCompareOperation mDepthComparison  = eDepthCompareOperation::ALWAYS;

        std::vector<sShaderData>          mShaderStages         = {};
        std::vector<sBufferLayoutElement> mInputLayout          = {};
        std::vector<sBufferLayoutElement> mInstancedInputLayout = {};
        std::vector<sPushConstantRange>   mPushConstants        = {};
        std::vector<sDescriptorSet>       mDescriptorLayout     = {};
    };
} // namespace SE::Graphics