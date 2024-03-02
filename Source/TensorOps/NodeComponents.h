/// @file   NodeComponents.h
///
/// @brief  Component definitions for computation graph nodes
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once
#include <variant>

#include "Core/Logging.h"
#include "Core/Definitions.h"

#include "Core/CUDA/Array/MemoryPool.h"
#include "Core/CUDA/Array/MultiTensor.h"
#include "Core/Entity/Collection.h"

#include "ScalarTypes.h"

namespace SE::TensorOps
{
    using namespace SE::Core;
    
    using MultiTensor  = SE::Cuda::multi_tensor_t;
    using MemoryBuffer = SE::Cuda::memory_buffer_t;
    using MemoryPool   = SE::Cuda::memory_pool_t;
    using sTensorShape = SE::Cuda::sTensorShape;

    using graph_node_t = SE::Core::Entity;

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

        graph_node_t GetControlledEntity()
        {
            return mEntity;
        };

      public:
        virtual void Initialize( graph_node_t aEntity )
        {
            mEntity = aEntity;
        }
        virtual void Run()
        {
        }

      private:
        graph_node_t mEntity;
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
        vector_t<_Ty> mValue = {}; //!< Values to upload

        sVectorValueComponent()                                = default;
        sVectorValueComponent( const sVectorValueComponent & ) = default;

        size_t Size()
        {
            return mValue.size();
        }
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
    using sScalarValueVectorComponent = sVectorValueComponent<scalar_value_t>;

    /// @brief sTypeComponent
    ///
    /// If added to an entity, indicates the numerical type of the elements in the node's output value.
    ///
    struct sTypeComponent
    {
        scalar_type_t mValue = scalar_type_t::UNKNOWN; //!< Type information

        sTypeComponent() = default;
        sTypeComponent( scalar_type_t a_Value )
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
        vector_t<graph_node_t> mOperands = {}; //!< List of entities that have to be run before the current entity.

        sOperandComponent() = default;
        sOperandComponent( vector_t<graph_node_t> const &aOpNodes )
        {
            mOperands = aOpNodes;
        };
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
        graph_node_t mLeft{};  //!< Lower bound. Should be a vector.
        graph_node_t mRight{}; //!< Upper bound. Should be a vector.
        graph_node_t mDelta{}; //!< Difference. Should be a vector.

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
        graph_node_t mLeftOperand;  //!< Left operand
        graph_node_t mRightOperand; //!< Right operand

        sBinaryOperationComponent()                                    = default;
        sBinaryOperationComponent( const sBinaryOperationComponent & ) = default;
    };

    struct sBroadcastInfoComponent
    {
        eBroadcastHint mBroadcastHint = eBroadcastHint::NONE; //!< How to broadcast the operation.

        graph_node_t   mBlockSizes;                                 //!< Block sizes
        uint32_t mMaxBlockSize = 0;                           //!< Maximum value of the `mBlockSizes` parameter
        graph_node_t   mBroadcastDimension;                         //!< Size of the broadcast dimension
        uint32_t mMaxBroadcastDimension = 0;                  //!< Maximum size of the broadcast dimension

        sBroadcastInfoComponent()                                  = default;
        sBroadcastInfoComponent( const sBroadcastInfoComponent & ) = default;
    };

    struct sNotOperationComponent
    {
        graph_node_t mOperand; //!< Left operand

        sNotOperationComponent()                                 = default;
        sNotOperationComponent( const sNotOperationComponent & ) = default;
    };

    struct sBitwiseNotOperationComponent
    {
        graph_node_t mOperand; //!< Left operand

        sBitwiseNotOperationComponent()                                        = default;
        sBitwiseNotOperationComponent( const sBitwiseNotOperationComponent & ) = default;
    };

    struct sInIntervalOperationComponent
    {
        graph_node_t mX;           //!< Value to test
        graph_node_t mLower;       //!< Interval lower bound
        graph_node_t mUpper;       //!< Interval upper bound
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
        graph_node_t mArray;       //!< @ref MultiTensor to repeat.
        graph_node_t mRepetitions; //!< Number of times the elements of the @ref MultiTensor should be repeated

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
        graph_node_t mArray;       //!< @ref MultiTensor to repeat.
        graph_node_t mRepetitions; //!< Number of times the elements of the @ref MultiTensor should be tiled

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
        graph_node_t mLeft{};         //!< Lower bound
        graph_node_t mRight{};        //!< Upper bound
        graph_node_t mSubdivisions{}; //!< Number of subdivisions

        sLinearSpaceComponent()                                = default;
        sLinearSpaceComponent( const sLinearSpaceComponent & ) = default;
    };

    struct sWhereOperationComponent
    {
        graph_node_t mCondition{};    //!< Lower bound
        graph_node_t mValueIfTrue{};  //!< Upper bound
        graph_node_t mValueIfFalse{}; //!< Number of subdivisions

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
        graph_node_t mA{}; //!< Left
        graph_node_t mB{}; //!< Right
        graph_node_t mT{}; //!< Coefficient

        sMixNodeComponent()                            = default;
        sMixNodeComponent( const sMixNodeComponent & ) = default;
    };

    /// @brief sScalarNodeComponent
    ///
    /// When added to an entity, this component marks said entity as containing a single scalar
    ///
    struct sScalarNodeComponent
    {
        scalar_value_t mValue = 0.0f;

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
        scalar_value_t mValue = 0.0f; //!< Value to initialize the @ref MultiTensor with.

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
        vector_t<scalar_value_t> mValue = {}; //!< Vector of values
        MemoryBuffer             mData;       //!< GPU representation of the data of values

        sVectorInitializerComponent() = default;

        template <typename _Ty>
        sVectorInitializerComponent( vector_t<_Ty> const &aValues )
        {
            uint32_t lVectorSize = aValues.size();
            mValue               = vector_t<scalar_value_t>( lVectorSize );

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
        vector_t<scalar_value_t> mValue = {}; //!< Vector of values

        sDataInitializerComponent() = default;

        template <typename _Ty>
        sDataInitializerComponent( vector_t<_Ty> const &aValues )
        {
            uint32_t lVectorSize = aValues.size();
            mValue               = vector_t<scalar_value_t>( lVectorSize );

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
        scalar_type_t mType = scalar_type_t::FLOAT32; //!< Type

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
        scalar_type_t mType = scalar_type_t::FLOAT32; //!< Type
        scalar_value_t mMean = 0.0f;                 //!< Expected value
        scalar_value_t mStd  = 1.0f;                 //!< Standard deviation

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

        sTensorShape &Shape()
        {
            return mShape;
        }
    };

    /// @brief sSample2DComponent
    ///
    /// When added to an entity, this component samples a list of textures with the provided coordinates.
    ///
    struct sSample2DComponent
    {
        graph_node_t mX{};        //!< X coordinates of the texture samples
        graph_node_t mY{};        //!< Y coordinates of the texture samples
        graph_node_t mTextures{}; //!< Textures to sample from

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
        scalar_type_t mOutputType = scalar_type_t::UINT32; //!< Integer type to use to engode the fixed point decimal numbers
        graph_node_t      mArray{};                          //!< Input tensor/
        graph_node_t      mScaling{};                        //!< Scaling factor.

        sToFixedPointNodeComponent()                                     = default;
        sToFixedPointNodeComponent( const sToFixedPointNodeComponent & ) = default;
    };

    /// @brief sAffineNodeComponent
    ///
    /// When added to an entity, this component defines a node that computes an affine transformation.
    ///
    struct sAffineNodeComponent
    {
        graph_node_t mA{}; //!< Coefficient
        graph_node_t mX{}; //!< Variable
        graph_node_t mB{}; //!< Translation

        sAffineNodeComponent()                               = default;
        sAffineNodeComponent( const sAffineNodeComponent & ) = default;
    };

    /// @brief sArraySliceNodeComponent
    struct sArraySliceNodeComponent
    {
        graph_node_t mArray;              //!< Tensor to transform
        graph_node_t mBegin;              //!< Lower index value for each layer
        graph_node_t mEnd;                //!< Upper index value for each layer

        graph_node_t mElementCount;       //!< Length of the last dimension of `mArray`
        graph_node_t mBlockSizes;         //!< Product of the lengths of the first rank-1 dimensions of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sArraySliceNodeComponent()                                   = default;
        sArraySliceNodeComponent( const sArraySliceNodeComponent & ) = default;
    };

    /// @brief sArraySummationNodeComponent
    struct sArraySummationNodeComponent
    {
        graph_node_t mArray;              //!< Tensor to transform
        graph_node_t mBegin;              //!< Lower index value for each layer
        graph_node_t mEnd;                //!< Upper index value for each layer

        graph_node_t mElementCount;       // Length of the last dimension of `mArray`
        graph_node_t mBlockSizes;         //!< Product of the lengths of the first rank-1 dimensions of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sArraySummationNodeComponent()                                       = default;
        sArraySummationNodeComponent( const sArraySummationNodeComponent & ) = default;
    };

    /// @brief sConv1DNodeComponent
    struct sConv1DNodeComponent
    {
        graph_node_t   mArray0;               //!< Tensor to transform
        graph_node_t   mElementCount0;        //!< Length of the last dimension of `mArray0`
        graph_node_t   mBlockSizes0;          //!< Product of the lengths of the first rank-1 dimensions of `mArray0`
        uint32_t mMaxElementCount0 = 0; //!< Maximum value of the `mElementCount0` parameter
        uint32_t mMaxBlockSize0    = 0; //!< Maximum value of the `aBlockSizes0` parameter

        graph_node_t   mArray1;               //!< Convolution kernel
        graph_node_t   mElementCount1;        //!< Length of the last dimension of `mArray1`
        graph_node_t   mBlockSizes1;          //!< Product of the lengths of the first rank-1 dimensions of `mArray1`
        uint32_t mMaxBlockSize1 = 0;    //!< Maximum value of the `aBlockSizes1` parameter

        sConv1DNodeComponent()                               = default;
        sConv1DNodeComponent( const sConv1DNodeComponent & ) = default;
    };

    /// @brief sCountTrueNodeComponent
    struct sCountTrueNodeComponent
    {
        graph_node_t mArray;              //!< Tensor to transform
        graph_node_t mBlockSizes;         //!< Product of the lengths of the first rank-1 dimensions of `mArray`
        graph_node_t mElementCount;       //!< Length of the last dimension of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sCountTrueNodeComponent()                                  = default;
        sCountTrueNodeComponent( const sCountTrueNodeComponent & ) = default;
    };

    /// @brief sCountNonZeroNodeComponent
    struct sCountNonZeroNodeComponent
    {
        graph_node_t mArray;              //!< Tensor to transform
        graph_node_t mBlockSizes;         //!< Product of the lengths of the first rank-1 dimensions of `mArray`
        graph_node_t mElementCount;       //!< Length of the last dimension of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sCountNonZeroNodeComponent()                                     = default;
        sCountNonZeroNodeComponent( const sCountNonZeroNodeComponent & ) = default;
    };

    /// @brief sCountZeroNodeComponent
    struct sCountZeroNodeComponent
    {
        graph_node_t mArray;              //!< Tensor to transform
        graph_node_t mBlockSizes;         //!< Product of the lengths of the first rank-1 dimensions of `mArray`
        graph_node_t mElementCount;       //!< Length of the last dimension of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sCountZeroNodeComponent()                                  = default;
        sCountZeroNodeComponent( const sCountZeroNodeComponent & ) = default;
    };

    /// @brief sFloorNodeComponent
    struct sFloorNodeComponent
    {
        graph_node_t mArray; //!< Tensor to transform

        sFloorNodeComponent()                              = default;
        sFloorNodeComponent( const sFloorNodeComponent & ) = default;
    };

    /// @brief sCeilNodeComponent
    struct sCeilNodeComponent
    {
        graph_node_t mArray; //!< Tensor to transform

        sCeilNodeComponent()                             = default;
        sCeilNodeComponent( const sCeilNodeComponent & ) = default;
    };

    /// @brief sAbsNodeComponent
    struct sAbsNodeComponent
    {
        graph_node_t mArray; //!< Tensor to transform

        sAbsNodeComponent()                            = default;
        sAbsNodeComponent( const sAbsNodeComponent & ) = default;
    };

    /// @brief sSqrtNodeComponent
    struct sSqrtNodeComponent
    {
        graph_node_t mArray; //!< Tensor to transform

        sSqrtNodeComponent()                             = default;
        sSqrtNodeComponent( const sSqrtNodeComponent & ) = default;
    };

    /// @brief sRoundNodeComponent
    struct sRoundNodeComponent
    {
        graph_node_t mArray; //!< Tensor to transform

        sRoundNodeComponent()                              = default;
        sRoundNodeComponent( const sRoundNodeComponent & ) = default;
    };

    /// @brief sDiffNodeComponent
    struct sDiffNodeComponent
    {
        graph_node_t   mArray; //!< Tensor to transform
        uint32_t mCount = 0;

        graph_node_t mBlockSizes;         //!< Product of the lengths of the first rank-1 dimensions of `mArray`
        graph_node_t mElementCount;       //!< Length of the last dimension of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sDiffNodeComponent()                             = default;
        sDiffNodeComponent( const sDiffNodeComponent & ) = default;
    };

    /// @brief sShiftNodeComponent
    struct sShiftNodeComponent
    {
        graph_node_t  mArray; //!< Tensor to transform
        int32_t mCount = 0;
        graph_node_t  mFillValue;

        graph_node_t mBlockSizes;         //!< Product of the lengths of the first rank-1 dimensions of `mArray`
        graph_node_t mElementCount;       //!< Length of the last dimension of `mArray`

        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `aBlockSizes` parameter

        sShiftNodeComponent()                              = default;
        sShiftNodeComponent( const sShiftNodeComponent & ) = default;
    };

    /// @brief sHCatNodeComponent
    struct sHCatNodeComponent
    {
        graph_node_t mArray0;             //!< First tensor to concatenate
        graph_node_t mArray1;             //!< Second tensor to concatenate

        graph_node_t   mBlockSizes;       //!< Product of the lengths of the first rank-1 dimensions of `mArray0`
        uint32_t mMaxBlockSize = 0; //!< Maximum value of the `mBlockSizes` parameter

        graph_node_t mElementCount0;      //!< Length of the last dimension of `mArray0`
        graph_node_t mElementCount1;      //!< Length of the last dimension of `mArray1`

        sHCatNodeComponent()                             = default;
        sHCatNodeComponent( const sHCatNodeComponent & ) = default;
    };
} // namespace SE::TensorOps
