#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "UI/Components/Table.h"
#include "UI/Layouts/BoxLayout.h"
#include "UI/UI.h"

#include "Enums.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;
    struct sMultiPulseEvent
    {
        int              mRowIndex;
        int              mEventIndex;
        int              mSubEventIndex;
        eEventType       mEventType;
        eEventStatus     mEventStatus;
        eReflectanceType mReflectanceType;
        double           mWavelength;
        double           mPosition;
        double           mCursorA;
        double           mCursorB;
        double           mSubCursorA;
        double           mSubCursorB;
        double           mLoss;
        double           mReflectance;
        double           mCurveLevel;
        double           mLossAtA;
        double           mLossAtB;
        double           mEstimatedCurveLevel;
        double           mEstimatedLoss;
        double           mEstimatedEndLevel;
        double           mEndNoiseLevel;
        double           mPeakPulseWidth;
        double           mPeakPower;
        double           mPeakSNR;
        bool             mConsiderAsPossibleEcho;
    };

    class UIMultiPulseEventTable : public UITable
    {
      public:
        UIMultiPulseEventTable();
        UIMultiPulseEventTable( UIMultiPulseEventTable const & ) = default;

        ~UIMultiPulseEventTable() = default;

        void Clear();
        void SetData( std::vector<sMultiPulseEvent> const &aData );
        void OnEventClicked( std::function<void( sMultiPulseEvent const & )> const &aOnRowClicked );

      protected:
        Ref<sFloat64Column> mPositionColumn;
        Ref<sFloat64Column> mLossColumn;
        Ref<sFloat64Column> mEstimatedLossColumn;
        Ref<sFloat64Column> mReflectanceColumn;
        Ref<sFloat64Column> mWavelengthColumn;
        Ref<sFloat64Column> mCursorAColumn;
        Ref<sFloat64Column> mCursorBColumn;
        Ref<sFloat64Column> mSubCursorAColumn;
        Ref<sFloat64Column> mSubCursorBColumn;
        Ref<sFloat64Column> mCurveLevelColumn;
        Ref<sFloat64Column> mLossAtAColumn;
        Ref<sFloat64Column> mLossAtBColumn;
        Ref<sFloat64Column> mEstimatedCurveLevelColumn;
        Ref<sFloat64Column> mEstimatedEndLevelColumn;
        Ref<sFloat64Column> mEndNoiseLevelColumn;
        Ref<sFloat64Column> mPeakPulseWidth;
        Ref<sFloat64Column> mPeakPower;
        Ref<sFloat64Column> mPeakSNR;

        std::vector<sEvent> mEventDataVector;

      private:
        std::function<void( sMultiPulseEvent const & )> mOnElementClicked;
        void                                            UIMultiPulseEventTable::DoSelectElement( uint32_t aRow );
    };
} // namespace SE::OtdrEditor