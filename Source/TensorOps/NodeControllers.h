/// @file   NodeControllers.h
///
/// @brief  Controller definitions for computation graph computation
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once

#include "Implementation/HelperMacros.h"
#include "NodeComponents.h"

namespace SE::TensorOps
{

    using namespace SE::Cuda;

    /// @struct sMultiTensorRunner
    ///
    /// Performs all necessary oeprations to initialize a constant multitensor.
    ///
    struct sMultiTensorRunner : public sGraphOperationController
    {
        void Run();
    };

    template <typename T>
    static inline void ResolveAndUpload( sDataInitializerComponent const &aComponent, MultiTensor const &aOut )
    {
        aOut.Upload( Private::Resolve<T>( aComponent.mValue ) );
    }

    template <typename T>
    static inline void ResolveAndUpload( sVectorInitializerComponent const &aComponent )
    {
        aComponent.mData.Upload( Private::Resolve<T>( aComponent.mValue ) );
    }

    template <typename T>
    static inline void ResolveAndUpload( MemoryBuffer &aData, vector_t<ScalarValue> const &aValue )
    {
        aData.Upload( Private::Resolve<T>( aValue ) );
    }

    /// @struct VectorRunner
    ///
    /// Performs all necessary oeprations to initialize a constant vector
    ///
    /// @tparam _Ty Type of vector elements/
    ///
    template <typename _Ty>
    struct VectorRunner : public sGraphOperationController
    {
        void Run()
        {
            auto &lData  = Get<sVectorBufferComponent>().mValue;
            auto &lValue = Get<sVectorValueComponent<_Ty>>().mValue;
            if constexpr( std::is_same<_Ty, ScalarValue>::value )
            {
                DISPATCH_BY_TYPE( TypeOf( lValue[0] ), ResolveAndUpload, ( lData, lValue ) );
            }
            else
            {
                lData.Upload( lValue );
            }
        }
    };

    /// @struct sBinaryOperationController
    ///
    /// Base class for binary operation nodes. This controller simply dispatches method calls to the proper implementation of binary
    /// operations.
    ///
    struct sBinaryOperationController : public sGraphOperationController
    {
        void Run();

        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight ) = 0;
        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                         sBroadcastInfoComponent &aBroadcast )                                                        = 0;

        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant ) = 0;
        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn ) = 0;
        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight ) = 0;
        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight ) = 0;
    };

    /// @brief sBinaryBooleanOperationController
    ///
    /// Base class for binary operation nodes. This controller simply dispatches method calls to the proper implementation of binary
    /// operations.
    ///
    struct sBinaryBooleanOperationController : public sGraphOperationController
    {
        void Run();

        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight ) = 0;
        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                         sBroadcastInfoComponent &aBroadcast )                                                        = 0;

        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant ) = 0;
        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn ) = 0;
        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight ) = 0;
        virtual void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight ) = 0;
    };

    /// @brief sAddOperationController
    ///
    /// Addition
    ///
    struct sAddOperationController : public sBinaryOperationController
    {
        sAddOperationController()                                  = default;
        sAddOperationController( const sAddOperationController & ) = default;

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                 sBroadcastInfoComponent &aBroadcast );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );
    };

    /// @struct sMultiplyOperationController
    ///
    /// Multiplication
    ///
    struct sMultiplyOperationController : public sBinaryOperationController
    {
      public:
        sMultiplyOperationController()                                       = default;
        sMultiplyOperationController( const sMultiplyOperationController & ) = default;

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                 sBroadcastInfoComponent &aBroadcast );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );
    };

    /// @struct sSubtractOperationController
    ///
    /// Subtraction
    ///
    struct sSubtractOperationController : public sBinaryOperationController
    {
      public:
        sSubtractOperationController()                                       = default;
        sSubtractOperationController( const sSubtractOperationController & ) = default;

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                 sBroadcastInfoComponent &aBroadcast );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );
    };

    /// @struct sDivideOperationController
    ///
    /// Division
    ///
    struct sDivideOperationController : public sBinaryOperationController
    {
      public:
        sDivideOperationController()                                     = default;
        sDivideOperationController( const sDivideOperationController & ) = default;

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                 sBroadcastInfoComponent &aBroadcast );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );
    };

    /// @brief sAndOperationController
    ///
    /// Conjunction
    ///
    struct sAndOperationController : public sBinaryOperationController
    {
      public:
        sAndOperationController()                                  = default;
        sAndOperationController( const sAndOperationController & ) = default;

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                 sBroadcastInfoComponent &aBroadcast );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );
    };

    /// @brief sOrOperationController
    ///
    /// Disjunction
    ///
    struct sOrOperationController : public sBinaryOperationController
    {
      public:
        sOrOperationController()                                 = default;
        sOrOperationController( const sOrOperationController & ) = default;

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                 sBroadcastInfoComponent &aBroadcast );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );
    };

    /// @brief sNotOperationController
    ///
    /// Negation
    ///
    struct sNotOperationController : public sGraphOperationController
    {
      public:
        sNotOperationController()                                  = default;
        sNotOperationController( const sNotOperationController & ) = default;

        void Run();
    };

    /// @brief sBitwiseAndOperationController
    ///
    /// Bitwise conjunction
    ///
    struct sBitwiseAndOperationController : public sBinaryOperationController
    {
      public:
        sBitwiseAndOperationController()                                         = default;
        sBitwiseAndOperationController( const sBitwiseAndOperationController & ) = default;

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                 sBroadcastInfoComponent &aBroadcast );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );
    };

    /// @brief sBitwiseOrOperationController
    ///
    /// Bitwise disjunction
    ///
    struct sBitwiseOrOperationController : public sBinaryOperationController
    {
      public:
        sBitwiseOrOperationController()                                        = default;
        sBitwiseOrOperationController( const sBitwiseOrOperationController & ) = default;

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                 sBroadcastInfoComponent &aBroadcast );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aIn, ScalarValue &aConstant );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aConstant, MultiTensor &aIn );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );
    };

    /// @brief sNotOperationController
    ///
    /// Bitwise negation
    ///
    struct sBitwiseNotOperationController : public sGraphOperationController
    {
      public:
        sBitwiseNotOperationController()                                         = default;
        sBitwiseNotOperationController( const sBitwiseNotOperationController & ) = default;

        void Run();
    };

    /// @brief sInIntervalOperationController
    ///
    /// Check whether the values in a given tensor lie between two given values
    ///
    struct sInIntervalOperationController : public sGraphOperationController
    {
      public:
        sInIntervalOperationController()                                         = default;
        sInIntervalOperationController( const sInIntervalOperationController & ) = default;

        void Run();
    };

    /// @brief sEqualOperationController
    ///
    /// Check whether two tensors are equal
    ///
    struct sEqualOperationController : public sBinaryBooleanOperationController
    {
      public:
        sEqualOperationController()                                    = default;
        sEqualOperationController( const sEqualOperationController & ) = default;

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                 sBroadcastInfoComponent &aBroadcast );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );
    };

    /// @brief sLessThanOperationController
    ///
    /// Strict inequality checking
    ///
    struct sLessThanOperationController : public sBinaryBooleanOperationController
    {
      public:
        sLessThanOperationController()                                       = default;
        sLessThanOperationController( const sLessThanOperationController & ) = default;

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                 sBroadcastInfoComponent &aBroadcast );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );
    };

    /// @brief sLessThanOrEqualOperationController
    ///
    /// Inequality checking
    ///
    struct sLessThanOrEqualOperationController : public sBinaryBooleanOperationController
    {
      public:
        sLessThanOrEqualOperationController()                                              = default;
        sLessThanOrEqualOperationController( const sLessThanOrEqualOperationController & ) = default;

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MultiTensor &aRight,
                 sBroadcastInfoComponent &aBroadcast );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, ScalarValue &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, ScalarValue &aLeft, MultiTensor &aRight );

        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MultiTensor &aLeft, MemoryBuffer &aRight );
        void Op( eScalarType aTensorElementType, MultiTensor &aOut, MemoryBuffer &aLeft, MultiTensor &aRight );
    };

    /// @brief sWhereOperationController
    ///
    /// Choose values according to boolean condition
    ///
    struct sWhereOperationController : public sGraphOperationController
    {
      public:
        sWhereOperationController()                                    = default;
        sWhereOperationController( const sWhereOperationController & ) = default;

        void Run();
    };

    /// @brief sARangeOperationController
    ///
    /// Addition
    ///
    struct sARangeOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @struct sArrayOperationController
    ///
    /// Controller for tiling and repetition of tensors/
    ///
    struct sArrayOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @struct sLinearSpaceOperationController
    ///
    /// Linear space
    ///
    struct sLinearSpaceOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @struct sMixOperationController
    ///
    /// Mix
    ///
    struct sMixOperationController : public sGraphOperationController
    {
        void OnCreate()
        {
        }
        void OnDestroy()
        {
        }
        void Run();
    };

    /// @struct sSample2DOperationController
    ///
    /// Texture sampling
    ///
    struct sSample2DOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @struct sToFixedPointOperationController
    ///
    /// Fixed point arithmetic conversion
    ///
    struct sToFixedPointOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @struct sAffineNodeController
    ///
    /// Affine transformation
    ///
    struct sAffineNodeController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Array slicing
    struct sArraySliceOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Array summation
    struct sArraySummationOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Count true elements
    struct sCountTrueOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Count non-zero elements
    struct sCountNonZeroOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Count zero elements
    struct sCountZeroOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief 1D convolution
    struct sConv1DOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Floor
    struct sFloorOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Ceiling
    struct sCeilOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Absolute value
    struct sAbsOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Square root
    struct sSqrtOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Round root
    struct sRoundOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Finite difference
    struct sDiffOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief Shift operator
    struct sShiftOperationController : public sGraphOperationController
    {
        void Run();
    };

    /// @brief HCat operator
    struct sHCatOperationController : public sGraphOperationController
    {
        void Run();
    };

} // namespace SE::TensorOps