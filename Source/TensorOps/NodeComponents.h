/// @file   NodeComponents.h
///
/// @brief  Component definitions for computation graph nodes
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once
#include <variant>

#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Core/EntityRegistry/Registry.h"
#include "Core/GPUResource/Array/MemoryPool.h"
#include "Core/GPUResource/Array/MultiTensor.h"

#include "ScalarTypes.h"

namespace SE::TensorOps
{

    using MultiTensor  = SE::Cuda::MultiTensor;
    using MemoryBuffer = SE::Cuda::MemoryBuffer;
    using MemoryPool   = SE::Cuda::MemoryPool;
    using sTensorShape = SE::Cuda::sTensorShape;

    using OpNode = SE::Core::Entity;

    /// @brief sGraphOperationController
    ///
    /// Base class for graph operation controller
    ///
    class sGraphOperationController
    {
      public:
        virtual ~sGraphOperationController() = default;

        template <typename T>
        T &Get()
        {
            return mEntity.Get<T>();
        }

        template <typename T>
        bool Has()
        {
            return mEntity.Has<T>();
        }

        OpNode GetControlledEntity() { return mEntity; };

      public:
        virtual void Initialize( OpNode aEntity ) { mEntity = aEntity; }
        virtual void Run() {}

      private:
        OpNode mEntity;
    };

    /// @brief sGraphOperationComponent
    ///
    /// This component marks the entity for execution by the tensor graph running system. In order
    /// for it wo do anything, it must be bound to a controller which provides the actual implementation
    /// of the tensor function.
    ///
    struct sGraphOperationComponent
    {
        sGraphOperationController *mControllerInstance = nullptr;

        std::function<sGraphOperationController *()>      mInstantiateController;
        std::function<void( sGraphOperationComponent * )> mDestroyController;

        template <typename T, typename... Args>
        void Bind( Args &&...args )
        {
            mInstantiateController = []() { return reinterpret_cast<sGraphOperationController *>( new T() ); };
            mDestroyController     = []( sGraphOperationComponent *nsc )
            {
                delete nsc->mControllerInstance;
                nsc->mControllerInstance = nullptr;
            };
        }
    };

    /// @brief sDoNotExpand
    ///
    /// Nodes tagged with this structure will not be added to the computation queue when the
    /// graph is run. Use this tag to avoid reprocessing a node whose values were already computed.
    ///
    struct sDoNotExpand
    {
        // This structure is intentionally empty.
    };

    /// @brief sAllocatedTag
    ///
    /// Nodes tagged with this structure already have their memory allocated in the pool
    ///
    struct sAllocatedTag
    {
        // This structure is intentionally empty.
    };

    /// @brief sVectorComponent
    ///
    /// Upload a vector of values to the GPU upon running
    ///
    template <typename _Ty>
    struct sVectorValueComponent
    {
        std::vector<_Ty> mValue = {}; //!< Values to upload

        sVectorValueComponent()                                = default;
        sVectorValueComponent( const sVectorValueComponent & ) = default;

        size_t Size() { return mValue.size(); }
    };

    struct sVectorBufferComponent
    {
        size_t       mSize = 0; //!< Size hint for the underlying memory buffer
        MemoryBuffer mValue{};  //!< GPU area where the values should be uploaded. This is typically allocated from a pool

        sVectorBufferComponent()                                 = default;
        sVectorBufferComponent( const sVectorBufferComponent & ) = default;
    };

    using sU32VectorComponent         = sVectorValueComponent<uint32_t>;
    using sF32VectorComponent         = sVectorValueComponent<float>;
    using sScalarValueVectorComponent = sVectorValueComponent<ScalarValue>;

    /// @brief sTypeComponent
    ///
    /// If added to an entity, indicates the numerical type of the elements in the node's output value.
    ///
    struct sTypeComponent
    {
        eScalarType mValue = eScalarType::UNKNOWN; //!< Type information

        sTypeComponent() = default;
        sTypeComponent( eScalarType a_Value )
            : mValue{ a_Value } {};
        sTypeComponent( const sTypeComponent & ) = default;
    };

    /// @brief sOperandComponent
    ///
    /// When added to an entity, this component holds a list of all the dependencies of said entity.  When running
    /// the graph, the elements listed in this component will be run first, then the current node will be run.
    ///
    struct sOperandComponent
    {
        std::vector<OpNode> mOperands = {}; //!< List of entities that have to be run before the current entity.

        sOperandComponent() = default;
        sOperandComponent( std::vector<OpNode> const &aOpNodes ) { mOperands = aOpNodes; };
        sOperandComponent( const sOperandComponent & ) = default;
    };

    /// @brief sARangeComponent
    ///
    /// When added to an entity, this component indicates that said entity should compute a range of values
    /// beween the values in `aLeft` and the values in `mRight` with a step of size `mDelta`.  All three of
    /// mLeft, mRight and Delta should be vectors of the same length
    ///
    struct sARangeComponent
    {
        OpNode mLeft{};  //!< Lower bound. Should be a vector.
        OpNode mRight{}; //!< Upper bound. Should be a vector.
        OpNode mDelta{}; //!< Difference. Should be a vector.

        sARangeComponent()                           = default;
        sARangeComponent( const sARangeComponent & ) = default;
    };

    /// @brief sBinaryOperationComponent
    ///
    /// When added to an entity, this component indicates that the entity represents a generic binary
    /// operation. The actual operation to perform depends on the operation controller.
    ///
    struct sBinaryOperationComponent
    {
        OpNode mLeftOperand;  //!< Left operand
        OpNode mRightOperand; //!< Right operand

        sBinaryOperationComponent()                                    = default;
        sBinaryOperationComponent( const sBinaryOperationComponent & ) = default;
    };

    struct sBroadcastInfoComponent
    {
        eBroadcastHint mBroadcastHint = eBroadcastHint::NONE; //!< How to broadcast the operation.

        OpNode   mBlockSizes;                //!< Block sizes
        uint32_t mMaxBlockSize = 0;          //!< Maximum value of the `mBlockSizes` parameter
        OpNode   mBroadcastDimension;        //!< Size of the broadcast dimension
        uint32_t mMaxBroadcastDimension = 0; //!< Maximum size of the broadcast dimension

        sBroadcastInfoComponent()                                  = default;
        sBroadcastInfoComponent( const sBroadcastInfoComponent & ) = default;
    };

    struct sNotOperationComponent
    {
        OpNode mOperand; //!< Left operand

        sNotOperationComponent()                                 = default;
        sNotOperationComponent( const sNotOperationComponent & ) = default;
    };

    struct sBitwiseNotOperationComponent
    {
        OpNode mOperand; //!< Left operand

        sBitwiseNotOperationComponent()                                        = default;
        sBitwiseNotOperationComponent( const sBitwiseNotOperationComponent & ) = default;
    };

    struct sInIntervalOperationComponent
    {
        OpNode mX;           //!< Value to test
        OpNode mLower;       //!< Interval lower bound
        OpNode mUpper;       //!< Interval upper bound
        bool   mStrictLower; //!< Use strict inequality for lower bound
        bool   mStrictUpper; //!< Use strict inequality for upper bound

        sInIntervalOperationComponent()                                        = default;
        sInIntervalOperationComponent( const sInIntervalOperationComponent & ) = default;
    };

    /// @brief sRepeatOperationComponent
    ///
    /// When added to an entity, this component indicates that the elements @ref MultiTensor contained in `mArray`
    /// should be repeated a number of times given by `mRepetitions`, which should contain a vector whose length
    /// should match the number of layers in the @ref MultiTensor contained in `mArray`.
    ///
    struct sRepeatOperationComponent
    {
        OpNode mArray;       //!< @ref MultiTensor to repeat.
        OpNode mRepetitions; //!< Number of times the elements of the @ref MultiTensor should be repeated

        sRepeatOperationComponent()                                    = default;
        sRepeatOperationComponent( const sRepeatOperationComponent & ) = default;
    };

    /// @brief sTileOperationComponent
    ///
    /// When added to an entity, this component indicates that the elements @ref MultiTensor contained in `mArray`
    /// should be tiled a number of times given by `mRepetitions`, which should contain a vector whose length
    /// should match the number of layers in the @ref MultiTensor contained in `mArray`.
    ///
    struct sTileOperationComponent
    {
        OpNode mArray;       //!< @ref MultiTensor to repeat.
        OpNode mRepetitions; //!< Number of times the elements of the @ref MultiTensor should be tiled

        sTileOperationComponent()                                  = default;
        sTileOperationComponent( const sTileOperationComponent & ) = default;
    };

    /// @brief sLinearSpaceComponent
    ///
    /// When added to an entity, this component indicates that the entity should compute a set of evenly spaced
    /// numbers between two bounds.
    ///
    struct sLinearSpaceComponent
    {
        OpNode mLeft{};         //!< Lower bound
        OpNode mRight{};        //!< Upper bound
        OpNode mSubdivisions{}; //!< Number of subdivisions

        sLinearSpaceComponent()                                = default;
        sLinearSpaceComponent( const sLinearSpaceComponent & ) = default;
    };

    struct sWhereOperationComponent
    {
        OpNode mCondition{};    //!< Lower bound
        OpNode mValueIfTrue{};  //!< Upper bound
        OpNode mValueIfFalse{}; //!< Number of subdivisions

        sWhereOperationComponent()                                   = default;
        sWhereOperationComponent( const sWhereOperationComponent & ) = default;
    };

    /// @brief sMixNodeComponent
    ///
    /// When added to an entity, this component indicated that said entity should compute the linear mix of
    /// the @ref MultiTensors represented by `mA` and `mB` with coefficient `t`.
    ///
    struct sMixNodeComponent
    {
        OpNode mA{}; //!< Left
        OpNode mB{}; //!< Right
        OpNode mT{}; //!< Coefficient

        sMixNodeComponent()                            = default;
        sMixNodeComponent( const sMixNodeComponent & ) = default;
    };

    /// @brief sScalarNodeComponent
    ///
    /// When added to an entity, this component marks said entity as containing a single scalar
    ///
    struct sScalarNodeComponent
    {
        ScalarValue mValue = 0.0f;

        sScalarNodeComponent()                               = default;
        sScalarNodeComponent( const sScalarNodeComponent & ) = default;
    };

    /// @brief sConstantValueInitializerComponent
    ///
    /// When added to an entity, this component marks the entity as containing a constant initializer
    /// for a @ref MultiTensor.
    ///
    struct sConstantValueInitializerComponent
    {
        ScalarValue mValue = 0.0f; //!< Value to initialize the @ref MultiTensor with.

        sConstantValueInitializerComponent() = default;
        template <typename _Ty>
        sConstantValueInitializerComponent( _Ty const &aValue )
        {
            mValue = aValue;
        }

        sConstantValueInitializerComponent( const sConstantValueInitializerComponent & ) = default;
    };

    /// @brief sVectorInitializerComponent
    ///
    /// When added to an entity, this component marks the entity as containing a constant initializer
    /// for a @ref MultiTensor with a different value for each layer.
    ///
    struct sVectorInitializerComponent
    {
        std::vector<ScalarValue> mValue = {}; //!< Vector of values
        MemoryBuffer             mData;       //!< GPU representation of the data of values

        sVectorInitializerComponent() = default;

        template <typename _Ty>
        sVectorInitializerComponent( std::vector<_Ty> const &aValues )
        {
            uint32_t lVectorSize = aValues.size();
            mValue               = std::vector<ScalarValue>( lVectorSize );

            for( uint32_t i = 0; i < lVectorSize; i++ )
            {
                mValue[i] = aValues[i];
            }
        }

        sVectorInitializerComponent( const sVectorInitializerComponent & ) = default;
    };

    /// @brief sDataInitializerComponent
    ///
    /// When added to an entity, this component marks the entity as containing a data initializer
    /// for a @ref MultiTensor containing data that should be copied to the MultiTensor upon
    /// initialization.
    ///

    struct sDataInitializerComponent
    {
        std::vector<ScalarValue> mValue = {}; //!< Vector of values

        sDataInitializerComponent() = default;

        template <typename _Ty>
        sDataInitializerComponent( std::vector<_Ty> const &aValues )
        {
            uint32_t lVectorSize = aValues.size();
            mValue               = std::vector<ScalarValue>( lVectorSize );

            for( uint32_t i = 0; i < lVectorSize; i++ )
            {
                mValue[i] = aValues[i];
            }
        }

        sDataInitializerComponent( const sDataInitializerComponent & ) = default;
    };

    /// @brief sRandomUniformInitializerComponent
    ///
    /// When added to an entity, this component indicates that the corresponding @ref MultiTensor
    /// should be filled with uniformly distributed random values
    ///
    struct sRandomUniformInitializerComponent
    {
        eScalarType mType = eScalarType::FLOAT32; //!< Type

        sRandomUniformInitializerComponent()                                             = default;
        sRandomUniformInitializerComponent( const sRandomUniformInitializerComponent & ) = default;
    };

    /// @brief sRandomNormalInitializerComponent
    ///
    /// When added to an entity, this component indicates that the corresponding @ref MultiTensor
    /// should be filled with normally distributed distributed random values
    ///
    struct sRandomNormalInitializerComponent
    {
        eScalarType mType = eScalarType::FLOAT32; //!< Type
        ScalarValue mMean = 0.0f;                 //!< Expected value
        ScalarValue mStd  = 1.0f;                 //!< Standard deviation

        sRandomNormalInitializerComponent()                                            = default;
        sRandomNormalInitializerComponent( const sRandomNormalInitializerComponent & ) = default;
    };

    /// @brief sMultiTensorComponent
    ///
    /// When added to an entity, this component attached a @ref MultiTensor, which can be
    /// filled eith with an initializer, or be used as the output of a tensor operation.
    ///
    struct sMultiTensorComponent
    {
        MultiTensor  mValue{}; //!< GPU buffer
        sTensorShape mShape{}; //!< Shape of the multitensor

        sMultiTensorComponent() = default;
        sMultiTensorComponent( MemoryPool &aMemoryPool, const sTensorShape &aShape )
            : mShape{ aShape }
        {
        }
        sMultiTensorComponent( MemoryPool &aMemoryPool, MemoryBuffer &aMemoryBuffer, const sTensorShape &aShape )
            : mShape{ aShape }

        {
        }
        sMultiTensorComponent( const sMultiTensorComponent & ) = default;

        sTensorShape &Shape() { return mShape; }
    };

    /// @brief sSample2DComponent
    ///
    /// When added to an entity, this component samples a list of textures with the provided coordinates.
    ///
    struct sSample2DComponent
    {
        OpNode mX{};        //!< X coordinates of the texture samples
        OpNode mY{};        //!< Y coordinates of the texture samples
        OpNode mTextures{}; //!< Textures to sample from

        sSample2DComponent()                             = default;
        sSample2DComponent( const sSample2DComponent & ) = default;
    };

    /// @brief sToFixedPointNodeComponent
    ///
    /// When added to an entity, this component defines a node that converts floating point numbers to fixed
    /// point decimal mnumbers encoded as integers.
    ///
    struct sToFixedPointNodeComponent
    {
        eScalarType mOutputType = eScalarType::UINT32; //!< Integer type to use to engode the fixed point decimal numbers
        OpNode      mArray{};                          //!< Input tensor/
        OpNode      mScaling{};                        //!< Scaling factor.

        sToFixedPointNodeComponent()                                     = default;
        sToFixedPointNodeComponent( const sToFixedPointNodeComponent & ) = default;
    };

    /// @brief sAffineNodeComponent
    ///
    /// When added to an entity, this component defines a node that computes an affine transformation.
    ///
    struct sAffineNodeComponent
    {
        OpNode mA{}; //!< Coefficient
        OpNode mX{}; //!< Variable
        OpNode mB{}; //!< Translation

        sAffineNodeComponent()                               = default;
        sAffineNodeComponent( const sAffineNodeComponent & ) = default;
    };

    /// @brief sArraySliceNodeComponent
    struct sArraySliceNodeComponent
    {
        OpNode mArray; //!< Tensor to transform
        OpNode mBegin; //!< Lower index value for each layer
        OpNode mEnd;   //!< Upper index value for each layer

        OpNode mElementCount; //!< Length of the last dimension of `mArray`
        OpNode mBlockSizes;   //!< Product of the lengths of the first rank-1 dimensions of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sArraySliceNodeComponent()                                   = default;
        sArraySliceNodeComponent( const sArraySliceNodeComponent & ) = default;
    };

    /// @brief sArraySummationNodeComponent
    struct sArraySummationNodeComponent
    {
        OpNode mArray; //!< Tensor to transform
        OpNode mBegin; //!< Lower index value for each layer
        OpNode mEnd;   //!< Upper index value for each layer

        OpNode mElementCount; // Length of the last dimension of `mArray`
        OpNode mBlockSizes;   //!< Product of the lengths of the first rank-1 dimensions of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sArraySummationNodeComponent()                                       = default;
        sArraySummationNodeComponent( const sArraySummationNodeComponent & ) = default;
    };

    /// @brief sConv1DNodeComponent
    struct sConv1DNodeComponent
    {
        OpNode   mArray0;               //!< Tensor to transform
        OpNode   mElementCount0;        //!< Length of the last dimension of `mArray0`
        OpNode   mBlockSizes0;          //!< Product of the lengths of the first rank-1 dimensions of `mArray0`
        uint32_t mMaxElementCount0 = 0; //!< Maximum value of the `mElementCount0` parameter
        uint32_t mMaxBlockSize0    = 0; //!< Maximum value of the `aBlockSizes0` parameter

        OpNode   mArray1;            //!< Convolution kernel
        OpNode   mElementCount1;     //!< Length of the last dimension of `mArray1`
        OpNode   mBlockSizes1;       //!< Product of the lengths of the first rank-1 dimensions of `mArray1`
        uint32_t mMaxBlockSize1 = 0; //!< Maximum value of the `aBlockSizes1` parameter

        sConv1DNodeComponent()                               = default;
        sConv1DNodeComponent( const sConv1DNodeComponent & ) = default;
    };

    /// @brief sCountTrueNodeComponent
    struct sCountTrueNodeComponent
    {
        OpNode mArray;        //!< Tensor to transform
        OpNode mBlockSizes;   //!< Product of the lengths of the first rank-1 dimensions of `mArray`
        OpNode mElementCount; //!< Length of the last dimension of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sCountTrueNodeComponent()                                  = default;
        sCountTrueNodeComponent( const sCountTrueNodeComponent & ) = default;
    };

    /// @brief sCountNonZeroNodeComponent
    struct sCountNonZeroNodeComponent
    {
        OpNode mArray;        //!< Tensor to transform
        OpNode mBlockSizes;   //!< Product of the lengths of the first rank-1 dimensions of `mArray`
        OpNode mElementCount; //!< Length of the last dimension of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sCountNonZeroNodeComponent()                                     = default;
        sCountNonZeroNodeComponent( const sCountNonZeroNodeComponent & ) = default;
    };

    /// @brief sCountZeroNodeComponent
    struct sCountZeroNodeComponent
    {
        OpNode mArray;        //!< Tensor to transform
        OpNode mBlockSizes;   //!< Product of the lengths of the first rank-1 dimensions of `mArray`
        OpNode mElementCount; //!< Length of the last dimension of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sCountZeroNodeComponent()                                  = default;
        sCountZeroNodeComponent( const sCountZeroNodeComponent & ) = default;
    };

    /// @brief sFloorNodeComponent
    struct sFloorNodeComponent
    {
        OpNode mArray; //!< Tensor to transform

        sFloorNodeComponent()                              = default;
        sFloorNodeComponent( const sFloorNodeComponent & ) = default;
    };

    /// @brief sCeilNodeComponent
    struct sCeilNodeComponent
    {
        OpNode mArray; //!< Tensor to transform

        sCeilNodeComponent()                             = default;
        sCeilNodeComponent( const sCeilNodeComponent & ) = default;
    };

    /// @brief sAbsNodeComponent
    struct sAbsNodeComponent
    {
        OpNode mArray; //!< Tensor to transform

        sAbsNodeComponent()                            = default;
        sAbsNodeComponent( const sAbsNodeComponent & ) = default;
    };

    /// @brief sSqrtNodeComponent
    struct sSqrtNodeComponent
    {
        OpNode mArray; //!< Tensor to transform

        sSqrtNodeComponent()                             = default;
        sSqrtNodeComponent( const sSqrtNodeComponent & ) = default;
    };

    /// @brief sRoundNodeComponent
    struct sRoundNodeComponent
    {
        OpNode mArray; //!< Tensor to transform

        sRoundNodeComponent()                              = default;
        sRoundNodeComponent( const sRoundNodeComponent & ) = default;
    };

    /// @brief sDiffNodeComponent
    struct sDiffNodeComponent
    {
        OpNode   mArray; //!< Tensor to transform
        uint32_t mCount = 0;

        OpNode mBlockSizes;   //!< Product of the lengths of the first rank-1 dimensions of `mArray`
        OpNode mElementCount; //!< Length of the last dimension of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sDiffNodeComponent()                             = default;
        sDiffNodeComponent( const sDiffNodeComponent & ) = default;
    };

    /// @brief sShiftNodeComponent
    struct sShiftNodeComponent
    {
        OpNode  mArray; //!< Tensor to transform
        int32_t mCount = 0;
        OpNode  mFillValue;

        OpNode mBlockSizes;   //!< Product of the lengths of the first rank-1 dimensions of `mArray`
        OpNode mElementCount; //!< Length of the last dimension of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sShiftNodeComponent()                              = default;
        sShiftNodeComponent( const sShiftNodeComponent & ) = default;
    };

    /// @brief sHCatNodeComponent
    struct sHCatNodeComponent
    {
        OpNode mArray0; //!< First tensor to concatenate
        OpNode mArray1; //!< Second tensor to concatenate

        OpNode   mBlockSizes;       //!< Product of the lengths of the first rank-1 dimensions of `mArray0`
        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `mBlockSizes` parameter

        OpNode mElementCount0; //!< Length of the last dimension of `mArray0`
        OpNode mElementCount1; //!< Length of the last dimension of `mArray1`

        sHCatNodeComponent()                             = default;
        sHCatNodeComponent( const sHCatNodeComponent & ) = default;
    };
} // namespace SE::TensorOps
