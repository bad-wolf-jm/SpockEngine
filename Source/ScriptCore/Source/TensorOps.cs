using System;

namespace SpockEngine
{
    public class sTensorShape
    {

    };

    public class Scope
    {

    };

    public struct sConstantValueInitializerComponent
    {

    };

    public struct sVectorInitializerComponent
    {

    };

    public struct sDataInitializerComponent
    {

    };

    public struct sRandomUniformInitializerComponent
    {

    };

    public struct sRandomNormalInitializerComponent
    {

    };

    public class TensorOps
    {
        OpNode MultiTensorValue(Scope aScope, sConstantValueInitializerComponent aInitializer, sTensorShape aShape);

        OpNode MultiTensorValue(Scope aScope, sVectorInitializerComponent aInitializer, sTensorShape aShape);

        OpNode MultiTensorValue(Scope aScope, sDataInitializerComponent aInitializer, sTensorShape aShape);

        OpNode MultiTensorValue(Scope aScope, sRandomUniformInitializerComponent aInitializer, sTensorShape aShape);

        OpNode MultiTensorValue(Scope aScope, sRandomNormalInitializerComponent aInitializer, sTensorShape aShape);

        template<typename _Ty> OpNode VectorValue(Scope aScope, std::vector<_Ty> aValue)
        {
            auto l_NewEntity = aScope.CreateNode();

            auto & l_Value = l_NewEntity.Add<sVectorComponent<_Ty>>();
            l_Value.mValue = aValue;
            l_Value.mData = aScope.mPool.Allocate(aValue.size() * sizeof(_Ty));

            if constexpr(std::is_same_v<_Ty, ScalarValue>)
            {
                l_NewEntity.Add<sTypeComponent>(TypeOf(aValue[0]));
            }

            l_NewEntity.Add<sGraphOperationComponent>().Bind<VectorRunner<_Ty>>();

            return l_NewEntity;
        }

        template<typename _Ty> OpNode ScalarVectorValue(Scope aScope, eScalarType aType, std::vector<_Ty> aValue)
        {
            uint32_t lSize = aValue.size();
            std::vector<ScalarValue> lValues(lSize );
            for (uint32_t i = 0; i < lSize; i++)
            {
                lValues[i] = aValue[i];
            }
            return VectorValue(aScope, lValues);
        }

        template<typename _Ty> OpNode ConstantScalarValue(Scope aScope, _Ty aValue)
        {
            auto l_NewEntity = aScope.CreateNode();

            auto & l_Value = l_NewEntity.Add<sScalarNodeComponent>();
            l_Value.mValue = aValue;

            l_NewEntity.Add<sTypeComponent>(TypeOf(l_Value.mValue));

            return l_NewEntity;
        }

        OpNode Add(Scope aScope, OpNode aLeft, OpNode aRight)
        {

        }

        OpNode Subtract(Scope aScope, OpNode aLeft, OpNode aRight)
        {

        }

        OpNode Divide(Scope aScope, OpNode aLeft, OpNode aRight)
        {

        }

        OpNode Multiply(Scope aScope, OpNode aLeft, OpNode aRight)
        {

        }

        OpNode And(Scope aScope, OpNode aLeft, OpNode aRight)
        {

        }

        OpNode Or(Scope aScope, OpNode aLeft, OpNode aRight)
        {

        }

        OpNode Not(Scope aScope, OpNode aOperand)
        {

        }

        OpNode BitwiseAnd(Scope aScope, OpNode aLeft, OpNode aRight)
        {

        }

        OpNode BitwiseOr(Scope aScope, OpNode aLeft, OpNode aRight)
        {

        }

        OpNode BitwiseNot(Scope aScope, OpNode aOperand)
        {

        }

        OpNode InInterval(Scope aScope, OpNode aX, OpNode aLower, OpNode aUpper, bool aStrictLower, bool aStrictUpper)
        {

        }

        OpNode Equal(Scope aScope, OpNode aX, OpNode aY)
        {

        }

        OpNode LessThan(Scope aScope, OpNode aX, OpNode aY)
        {

        }

        OpNode LessThanOrEqual(Scope aScope, OpNode aX, OpNode aY)
        {

        }

        OpNode GreaterThan(Scope aScope, OpNode aX, OpNode aY)
        {

        }

        OpNode GreaterThanOrEqual(Scope aScope, OpNode aX, OpNode aY)
        {

        }

        OpNode Where(Scope aScope, OpNode aCondition, OpNode aValueIfTrue, OpNode aValueIfFalse)
        {

        }

        OpNode Mix(Scope aScope, OpNode aA, OpNode aB, OpNode aT)
        {

        }

        OpNode AffineTransform(Scope aScope, OpNode aA, OpNode aX, OpNode aB)
        {

        }

        OpNode ARange(Scope aScope, OpNode aLeft, OpNode aRight, OpNode aDelta)
        {

        }

        OpNode LinearSpace(Scope aScope, OpNode aLeft, OpNode aRight, OpNode aSubdivisions)
        {

        }

        OpNode Repeat(Scope aScope, OpNode aArray, OpNode aRepetitions)
        {

        }

        OpNode Tile(Scope aScope, OpNode aArray, OpNode aRepetitions)
        {

        }

        OpNode Sample2D(Scope aScope, OpNode aX, OpNode aY, OpNode aTextures)
        {

        }

        OpNode ToFixedPoint(Scope aScope, eScalarType aOutputType, OpNode aArray, OpNode aScaling)
        {

        }

        OpNode Collapse(Scope aScope, OpNode aArray)
        {

        }

        OpNode Expand(Scope aScope, OpNode aArray)
        {

        }

        OpNode Reshape(Scope aScope, OpNode aArray, sTensorShape &aNewShape )
        {

        }

        OpNode Relayout(Scope aScope, OpNode aArray, sTensorShape &aNewLayout )
        {

        }

        OpNode Flatten(Scope aScope, OpNode aArray)
        {

        }

        OpNode Slice(Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd)
        {

        }

        OpNode Summation(Scope aScope, OpNode aArray)
        {

        }

        OpNode Summation(Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd)
        {

        }

        OpNode CountTrue(Scope aScope, OpNode aArray)
        {

        }

        OpNode CountNonZero(Scope aScope, OpNode aArray)
        {

        }

        OpNode CountZero(Scope aScope, OpNode aArray)
        {

        }

        OpNode Floor(Scope aScope, OpNode aArray)
        {

        }

        OpNode Ceil(Scope aScope, OpNode aArray)
        {

        }

        OpNode Abs(Scope aScope, OpNode aArray)
        {

        }

        OpNode Sqrt(Scope aScope, OpNode aArray)
        {

        }

        OpNode Round(Scope aScope, OpNode aArray)
        {

        }

        OpNode Diff(Scope aScope, OpNode aArray, uint32_t aCount)
        {

        }

        OpNode Shift(Scope aScope, OpNode aArray, int32_t aCount, OpNode aFillValue)
        {

        }

        OpNode Conv1D(Scope aScope, OpNode aArray0, OpNode aArray1)
        {

        }

        OpNode HCat(Scope aScope, OpNode aArray0, OpNode aArray1)
        {

        }
    };
}