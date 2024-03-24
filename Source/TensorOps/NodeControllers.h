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

    using namespace SE::Core;
    using namespace SE::Cuda;

    /// @struct sMultiTensorRunner
    ///
    /// Performs all necessary oeprations to initialize a constant multitensor.
    ///
    struct sMultiTensorRunner : public graph_operation_controller_t
    {
        void Run();
    };

    template <typename T>
    static inline void ResolveAndUpload( data_initializer_t const &aComponent, multi_tensor_t const &aOut )
    {
        aOut.Upload( Private::Resolve<T>( aComponent.mValue ) );
    }

    template <typename T>
    static inline void ResolveAndUpload( vector_initializer_t const &aComponent )
    {
        aComponent.mData.Upload( Private::Resolve<T>( aComponent.mValue ) );
    }

    template <typename T>
    static inline void ResolveAndUpload( memory_buffer_t &aData, vector_t<scalar_value_t> const &aValue )
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
    struct VectorRunner : public graph_operation_controller_t
    {
        void Run()
        {
            auto &lData  = Get<vector_buffer_t>().mValue;
            auto &lValue = Get<vector_value_t<_Ty>>().mValue;
            if constexpr( std::is_same<_Ty, scalar_value_t>::value )
            {
                DISPATCH_BY_TYPE( type_of( lValue[0] ), ResolveAndUpload, ( lData, lValue ) );
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
    struct sBinaryOperationController : public graph_operation_controller_t
    {
        void Run();

        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight ) = 0;
        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                         broadcast_info_t &aBroadcast )                                                        = 0;

        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant ) = 0;
        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn ) = 0;
        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight ) = 0;
        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight ) = 0;
    };

    /// @brief sBinaryBooleanOperationController
    ///
    /// Base class for binary operation nodes. This controller simply dispatches method calls to the proper implementation of binary
    /// operations.
    ///
    struct sBinaryBooleanOperationController : public graph_operation_controller_t
    {
        void Run();

        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight ) = 0;
        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                         broadcast_info_t &aBroadcast )                                                        = 0;

        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant ) = 0;
        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn ) = 0;
        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight ) = 0;
        virtual void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight ) = 0;
    };

    /// @brief sAddOperationController
    ///
    /// Addition
    ///
    struct sAddOperationController : public sBinaryOperationController
    {
        sAddOperationController()                                  = default;
        sAddOperationController( const sAddOperationController & ) = default;

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                 broadcast_info_t &aBroadcast );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );
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

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                 broadcast_info_t &aBroadcast );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );
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

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                 broadcast_info_t &aBroadcast );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );
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

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                 broadcast_info_t &aBroadcast );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );
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

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                 broadcast_info_t &aBroadcast );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );
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

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                 broadcast_info_t &aBroadcast );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );
    };

    /// @brief sNotOperationController
    ///
    /// Negation
    ///
    struct sNotOperationController : public graph_operation_controller_t
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

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                 broadcast_info_t &aBroadcast );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );
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

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                 broadcast_info_t &aBroadcast );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aIn, scalar_value_t &aConstant );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aConstant, multi_tensor_t &aIn );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );
    };

    /// @brief sNotOperationController
    ///
    /// Bitwise negation
    ///
    struct sBitwiseNotOperationController : public graph_operation_controller_t
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
    struct sInIntervalOperationController : public graph_operation_controller_t
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

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                 broadcast_info_t &aBroadcast );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );
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

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                 broadcast_info_t &aBroadcast );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );
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

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, multi_tensor_t &aRight,
                 broadcast_info_t &aBroadcast );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, scalar_value_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, scalar_value_t &aLeft, multi_tensor_t &aRight );

        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, multi_tensor_t &aLeft, memory_buffer_t &aRight );
        void Op( scalar_type_t aTensorElementType, multi_tensor_t &aOut, memory_buffer_t &aLeft, multi_tensor_t &aRight );
    };

    /// @brief sWhereOperationController
    ///
    /// Choose values according to boolean condition
    ///
    struct sWhereOperationController : public graph_operation_controller_t
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
    struct sARangeOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @struct sArrayOperationController
    ///
    /// Controller for tiling and repetition of tensors/
    ///
    struct sArrayOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @struct sLinearSpaceOperationController
    ///
    /// Linear space
    ///
    struct sLinearSpaceOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @struct sMixOperationController
    ///
    /// Mix
    ///
    struct sMixOperationController : public graph_operation_controller_t
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
    struct sSample2DOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @struct sToFixedPointOperationController
    ///
    /// Fixed point arithmetic conversion
    ///
    struct sToFixedPointOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @struct sAffineNodeController
    ///
    /// Affine transformation
    ///
    struct sAffineNodeController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Array slicing
    struct sArraySliceOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Array summation
    struct sArraySummationOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Count true elements
    struct sCountTrueOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Count non-zero elements
    struct sCountNonZeroOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Count zero elements
    struct sCountZeroOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief 1D convolution
    struct sConv1DOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Floor
    struct sFloorOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Ceiling
    struct sCeilOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Absolute value
    struct sAbsOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Square root
    struct sSqrtOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Round root
    struct sRoundOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Finite difference
    struct sDiffOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief Shift operator
    struct sShiftOperationController : public graph_operation_controller_t
    {
        void Run();
    };

    /// @brief HCat operator
    struct sHCatOperationController : public graph_operation_controller_t
    {
        void Run();
    };

} // namespace SE::TensorOps