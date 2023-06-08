using System;
using System.Runtime.CompilerServices;

using SpockEngine.Math;

namespace SpockEngine
{
    public class UIZLayout : UIComponent
    {
        public UIZLayout() : base(UIZLayout_Create()) { }

        ~UIZLayout() { UIZLayout_Destroy(mInstance); }

        public void Add(UIComponent aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment)
        {
            UIZLayout_AddAlignedNonFixed(mInstance, aChild.Instance, aExpand, aFill, aHAlignment, aVAlignment);
        }

        public void Add(UIComponent aChild, bool aExpand, bool aFill)
        {
            UIZLayout_AddNonAlignedNonFixed(mInstance, aChild.Instance, aExpand, aFill);
        }

        public void Add(UIComponent aChild, SpockEngine.Math.vec2 aSize, SpockEngine.Math.vec2 aPosition, bool aExpand, bool aFill)
        {
            UIZLayout_AddNonAlignedFixed(mInstance, aChild.Instance, aSize, aPosition, aExpand, aFill);
        }

        public void Add(UIComponent aChild, SpockEngine.Math.vec2 aSize, SpockEngine.Math.vec2 aPosition, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment)
        {
            UIZLayout_AddAlignedFixed(mInstance, aChild.Instance, aSize, aPosition, aExpand, aFill, aHAlignment, aVAlignment);
        }
    }
}